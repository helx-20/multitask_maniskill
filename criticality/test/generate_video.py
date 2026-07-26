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

from examples.baselines.ppo.task_registry import TASKS, by_env_id, by_task_id
from criticality.test.maniskill_ordinary_nade import make_env
from mani_skill.utils.wrappers.record import RecordEpisode

# import warnings
# warnings.filterwarnings("ignore", message=".*UserWarning.*", category=UserWarning)

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # resolve task: prefer --task_id when provided
    if getattr(args, "task_id", None) is not None:
        spec = by_task_id(int(args.task_id))
        args.env_id = spec.env_id
    else:
        spec = by_env_id(args.env_id)
    print(f"[*] Initializing env: {spec.env_id} (task_id={spec.task_id})", flush=True)
    # force rgb_array render mode for video generation
    args.render_mode = "rgb_array"
    env = make_env(args)

    env = RecordEpisode(env, output_dir=args.save_video_dir, save_trajectory=False, save_video_trigger=lambda x: True, max_steps_per_video=getattr(args, "num_steps", 1000), video_fps=5)

    print(f"[*] Loading policy: {args.checkpoint}", flush=True)
    mt_task_id = None
    from examples.baselines.ppo.multitask_agent import MultiTaskAgent
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
    mt_task_id = spec.task_id
    agent.eval()
    
    if args.log_std is not None:
        print(f"[*] Injecting policy variance log_std = {args.log_std}")
        with torch.no_grad():
            agent.actor_logstd.data.fill_(args.log_std)
    
    action_low = torch.tensor(env.get_wrapper_attr("single_action_space").low, device=device, dtype=torch.float32)
    action_high = torch.tensor(env.get_wrapper_attr("single_action_space").high, device=device, dtype=torch.float32)
    
    crashes = []
    weighted_crashes = []

    for ep in range(args.n):
        obs, info = env.reset(seed=args.worker_id * args.n + ep)
        success_once = False
        done = False
        steps = 0
        
        while steps < 100 and (not done or args.ignore_terminations):
            steps += 1
            obs_tensor = torch.as_tensor(obs).to(device)
            if obs_tensor.ndim == 1: obs_tensor = obs_tensor.unsqueeze(0)

            with torch.no_grad():
                action = agent.get_action(obs_tensor, deterministic=True)
            
            action = torch.clamp(action, action_low, action_high)

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Signal extraction logic
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

            obs = next_obs
            done = bool(terminated) or bool(truncated)

        # Episode settlement
        is_crash = 1 if not success_once else 0
        total_weight = info.get("criticality_info", {}).get("total_weight", 1.0)
        
        crashes.append(is_crash)
        weighted_crashes.append(is_crash * total_weight)

        if is_crash == 1:
            print(f"[Crash] Ep: {ep} | W: {total_weight:.4e}", flush=True)

        # if (ep + 1) % 10 == 0:
        #     elapsed = time.time() - start_time
        #     mu_hat = np.mean(weighted_crashes)
        #     n_samples = len(weighted_crashes)
        #     rhf = 0.0
        #     if n_samples > 1 and mu_hat > 1e-15:
        #         sigma_hat = np.std(weighted_crashes, ddof=1) 
        #         rhf = (z_score * sigma_hat) / (np.sqrt(n_samples) * mu_hat)
        #     print(f"Ep: {ep+1}/{args.n} | Crash Num: {sum(crashes)} | Crash Rate: {mu_hat:.4e} | RHF: {rhf:.3f}", flush=True)

    print(f"[*] Done! Final Crash Rate: {np.mean(weighted_crashes):.6e}", flush=True)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--task_id', type=int, default=3,
                        help="0=push, 1=pick, 2=stack, 3=peg. Overrides --env_id when set.")
    parser.add_argument('--checkpoint', type=str, default='examples/baselines/ppo/runs/multitask__ppo_multitask__1__1780644413/multitask_final_ckpt.pt')
    parser.add_argument('--criticality_ckpt', type=str, default='criticality/stage1/model/stage1_criticality_best_1_update.pt')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--n', type=int, default=100)
    
    parser.add_argument('--force_mag', type=float, default=1.0)
    parser.add_argument('--force_prob', type=float, default=1.0)
    parser.add_argument('--grid_size', type=int, default=11)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument("--obs_mode", type=str, default="state")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--sim_backend", type=str, default="physx_cpu")
    parser.add_argument('--nade', action='store_true', default=False)
    parser.add_argument('--criticality_threshold', type=float, default=0.5, help="Threshold for applying disturbance in NADE")
    parser.add_argument('--weight_threshold', type=float, default=1e-2)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--save_video_dir', type=str, default='criticality/test/videos_origin')
    parser.add_argument('--ignore_terminations', type=bool, default=True)
    
    parser.add_argument('--log_std', type=float, default=None, help="Initial log_std for data collection noise")
    
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_video_dir, exist_ok=True)
    np.random.seed(args.worker_id)
    torch.manual_seed(args.worker_id)                  
    torch.cuda.manual_seed(args.worker_id)             
    torch.cuda.manual_seed_all(args.worker_id)          
    torch.backends.cudnn.deterministic = True

    main(args)