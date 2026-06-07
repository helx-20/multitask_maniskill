import argparse
import os
import sys
import numpy as np
import torch
import gymnasium as gym
import time
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.baselines.ppo.task_registry import TASKS, by_env_id, by_task_id, num_tasks, obs_dims as registry_obs_dims
from criticality.test.maniskill_ordinary_nade import make_env

# import warnings
# warnings.filterwarnings("ignore", message=".*UserWarning.*", category=UserWarning)

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _resolve_task(args):
    if getattr(args, "task_id", None) is not None:
        spec = by_task_id(int(args.task_id))
        args.env_id = spec.env_id  # keep both in sync
    else:
        spec = by_env_id(args.env_id)
    return spec


def _load_agent(args, env, device):
    """Load a single-task `Agent` (default)`."""
    from examples.baselines.ppo.multitask_agent import MultiTaskAgent
    # we need obs_dims for all tasks; probe missing ones at runtime via env
    spec = _resolve_task(args)
    obs_dims_list = []
    for s in TASKS:
        if s.task_id == spec.task_id:
            obs_dims_list.append(int(np.prod(env.single_observation_space.shape)))
        else:
            obs_dims_list.append(s.obs_dim or 1)
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = MultiTaskAgent(input_dim=48, action_dim=action_dim).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    agent.load_state_dict(sd)
    agent.eval()
    return agent, spec.task_id


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # determine tasks to run: single or all
    if args.all_tasks:
        specs = TASKS
    else:
        specs = [ _resolve_task(args) ]

    print(f"[*] Will run tests for tasks: {[s.task_id for s in specs]}")
    for spec in specs:
        # ensure args reflect current task
        args.env_id = spec.env_id
        args.task_id = spec.task_id

        print(f"[*] 初始化环境: {spec.env_id} (task_id={spec.task_id})", flush=True)
        env = make_env(args)

        print(f"[*] 加载策略: {args.checkpoint}", flush=True)
        agent, mt_task_id = _load_agent(args, env, device)
        
        if args.log_std is not None:
            print(f"[*] 注入策略方差 log_std = {args.log_std}")
            with torch.no_grad():
                agent.actor_logstd.data.fill_(args.log_std)
        
        action_low = torch.tensor(env.get_wrapper_attr("single_action_space").low, device=device, dtype=torch.float32)
        action_high = torch.tensor(env.get_wrapper_attr("single_action_space").high, device=device, dtype=torch.float32)
        
        crashes = []
        weighted_crashes = []
        z_score = norm.ppf(1 - 0.1 / 2) 
        
        buffer = {'obs': [], 'actions': [], 'weights': [], 'rewards': [], 'dones': [], 'log_probs': []}
        
        start_time = time.time()

        # if task_id == 1 (pick), increase episodes because crash rate is very low
        n_episodes = args.n
        for ep in range(n_episodes):
            obs, info = env.reset(seed=args.worker_id * n_episodes + ep + spec.task_id)
            ep_obs, ep_acts, ep_weights, ep_log_probs = [], [], [], []
            success_once = False
            done = False
            
            while not done:
                obs_tensor = torch.as_tensor(obs).to(device)
                if obs_tensor.ndim == 1: obs_tensor = obs_tensor.unsqueeze(0)

                with torch.no_grad():
                    # use sampled action + log_prob when training_out requested
                    if args.training_out is not None:
                        action, log_prob, _, _ = agent.get_action_and_value(obs_tensor)
                    else:
                        action = agent.get_action(obs_tensor, deterministic=True)
                
                action = torch.clamp(action, action_low, action_high)
                
                # record
                if args.training_out is not None:
                    ep_obs.append(to_np(obs).flatten())
                    ep_acts.append(to_np(action).flatten())
                    ep_log_probs.append(to_np(log_prob).flatten())

                next_obs, reward, terminated, truncated, info = env.step(action)

                # success extraction
                current_success = False
                if info.get("_final_info", False):
                    fi = info.get("final_info", {})
                    current_success = fi.get("episode", {}).get("success_once", False)
                else:
                    current_success = info.get("success", False)
                
                if hasattr(current_success, "item"): current_success = bool(current_success.item())
                elif isinstance(current_success, np.ndarray): current_success = bool(current_success.any())
                else: current_success = bool(current_success)
                
                success_once = success_once or current_success

                sw = info.get("criticality_info", {}).get("weight", 1.0)
                ep_weights.append(float(sw))

                obs = next_obs
                done = bool(terminated) or bool(truncated)

            # end episode
            is_crash = 1 if not success_once else 0
            total_weight = info.get("criticality_info", {}).get("total_weight", 1.0)
            crashes.append(is_crash)
            weighted_crashes.append(is_crash * total_weight)

            if args.training_out is not None:
                buffer['obs'].extend(ep_obs)
                buffer['actions'].extend(ep_acts)
                buffer['weights'].extend(ep_weights)
                buffer['log_probs'].extend(ep_log_probs)
                rews = [0.0] * len(ep_obs)
                if success_once: rews[-1] = 1.0
                buffer['rewards'].extend(rews)
                dns = [False] * len(ep_obs); dns[-1] = True
                buffer['dones'].extend(dns)

            if is_crash == 1:
                print(f"[Crash] task={spec.short_name} Ep: {ep} | W: {total_weight:.4e}", flush=True)

            if (ep + 1) % 10 == 0:
                elapsed = time.time() - start_time
                mu_hat = np.mean(weighted_crashes)
                n_samples = len(weighted_crashes)
                rhf = 0.0
                if n_samples > 1 and mu_hat > 1e-15:
                    sigma_hat = np.std(weighted_crashes, ddof=1)
                    rhf = (z_score * sigma_hat) / (np.sqrt(n_samples) * mu_hat)
                print(f"task={spec.short_name} Ep: {ep+1}/{n_episodes} | Crash Num: {sum(crashes)} | Crash Rate: {mu_hat:.4e} | RHF: {rhf:.3f}", flush=True)

            if (ep + 1) % args.save_freq == 0:
                if args.training_out is None:
                    mode = "nade" if args.nade else "nde"
                    save_path = os.path.join(args.save_dir, f"{mode}_{spec.short_name}_{args.worker_id}.npy")
                    np.save(save_path, np.array(weighted_crashes))
                else:
                    save_path = os.path.join(args.training_out, f"training_{spec.short_name}_{args.worker_id}.npy")
                    np.save(save_path, np.array(buffer))

        # end task
        print(f"[*] 完成 task={spec.short_name} | 最终 Crash Rate: {np.mean(weighted_crashes):.6e}", flush=True)
        env.close()
        if args.training_out is None:
            mode = "nade" if args.nade else "nde"
            save_path = os.path.join(args.save_dir, f"{mode}_{spec.short_name}_{args.worker_id}.npy")
            np.save(save_path, np.array(weighted_crashes))
        else:
            save_path = os.path.join(args.training_out, f"training_{spec.short_name}_{args.worker_id}.npy")
            np.save(save_path, np.array(buffer))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--task_id', type=int, default=1,
                        help="0=push, 1=pick, 2=stack, 3=peg. Overrides --env_id when set.")
    parser.add_argument('--all_tasks', default=False,
                        help='Run the test for all tasks (each task n episodes)')
    parser.add_argument('--checkpoint', type=str, default='examples/baselines/ppo/runs/multitask__ppo_multitask__1__1780644413/multitask_final_ckpt.pt')
    # parser.add_argument('--checkpoint', type=str, default='training/models/round4/offline_model_best.pt')
    parser.add_argument('--criticality_ckpt', type=str, default='criticality/stage1/model/stage1_criticality_best_1.pt')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='./test_results')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--force_mag', type=float, default=1.0)
    parser.add_argument('--force_prob', type=float, default=1.0)
    parser.add_argument('--grid_size', type=int, default=11)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument("--obs_mode", type=str, default="state")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--sim_backend", type=str, default="physx_cpu")
    parser.add_argument('--nade', action='store_true', default=False)
    parser.add_argument('--criticality_threshold', type=float, default=0.5, help="Threshold for applying disturbance in NADE")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon for epsilon-greedy exploration in NADE")
    parser.add_argument('--weight_threshold', type=float, default=1e-3, help="Threshold for cumulative importance weight in NADE")
    parser.add_argument('--training_out', type=str, default=None)
    
    parser.add_argument('--log_std', type=float, default=None, help="Initial log_std for data collection noise")
    
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.training_out:
        os.makedirs(args.training_out, exist_ok=True)
    else:
        np.random.seed(args.worker_id)
        torch.manual_seed(args.worker_id)                  
        torch.cuda.manual_seed(args.worker_id)             
        torch.cuda.manual_seed_all(args.worker_id)          
        torch.backends.cudnn.deterministic = True

    main(args)