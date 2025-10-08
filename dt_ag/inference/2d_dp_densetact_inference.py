#!/usr/bin/env python3
import sys, pathlib, os

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEBUG_DIR = SCRIPT_DIR / "2d_dp_debug_baseline_alex"

DT_AG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DT_AG_ROOT))

LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[2] / "universal_manipulation_interface"
assert LOCAL_ROOT.is_dir(), f"Can't find code at {LOCAL_ROOT}"
sys.path.insert(0, str(LOCAL_ROOT))
os.chdir(str(LOCAL_ROOT))

import diffusion_policy
print("Loading diffusion_policy from:", diffusion_policy.__file__)
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import rclpy
from rclpy.node import Node
import threading
import numpy as np
import torch
import time
import diffusers
import hydra
import dill
import pygame
import multiprocessing
from multiprocessing import Process, Manager, Queue

from utils.inference_utils import (
    InferenceUtils,
    save_data,
    log_policy,
    log_inference_timing,
    apply_vision_blackout
)


def load_policy(model_path):
    path = pathlib.Path(model_path)
    payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

    cfg = payload['cfg']
    print(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16

    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)

    device = torch.device('cuda')
    policy.eval().to(device)

    return policy, cfg

# ----------------------------------------------------------------------
# Policy node
# ----------------------------------------------------------------------
class PolicyNode3D(InferenceUtils, Node):
    def __init__(self, shared_obs, action_queue, start_time, model_path: str):
        super().__init__('Policy_Node')
        self.init_policy_node(model_path, shared_obs, action_queue, start_time, sync_queue_size=100, sync_slop=0.05, inference_rate=30)

    def timer_callback(self):
        """
        Timer callback that checks for scheduled actions and executes them when their time arrives.
        Also handles clearing pending actions when new inference results arrive.
        Based on the scheduling approach from the reference code.
        """
        current_time = time.monotonic()
        
        # Drain any new items from the action queue 
        while not self.action_queue.empty():
            item = self.action_queue.get()
            
            if (isinstance(item, tuple) and len(item) == 2 and 
                isinstance(item[0], str) and item[0] == "CLEAR_PENDING"):
                clear_time = item[1]
                num_cleared = len(self.pending_actions)
                self.pending_actions.clear()
                print(f"🗑️  Cleared {num_cleared} pending actions at {clear_time:.3f}s")
            else:
                try:
                    act, ts = item
                    self.pending_actions.append((act, ts))
                    print(f"📥 Received new action scheduled for {ts:.3f}s")
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Received unexpected item in action queue: {type(item)}, {e}")
                    continue
        
        # Update execution status
        if not self.pending_actions:
            self.shared_obs["exec_done"] = True
            return
        else:
            self.shared_obs["exec_done"] = False
        
        # Determine which pending actions should be published now
        actions_to_publish = []
        remaining_actions = []
        
        for act, ts in self.pending_actions:
            if current_time >= ts and not self.paused:
                actions_to_publish.append((act, ts))
                print(f"⚡ Executing action scheduled for {ts:.3f}s (current: {current_time:.3f}s, delay: {current_time - ts:.3f}s)")
            else:
                remaining_actions.append((act, ts))
        
        # Update pending actions list
        self.pending_actions = remaining_actions
        
        # Publish all actions that are scheduled for now
        for action, scheduled_time in actions_to_publish:
            if not self.paused:
                ee_msg, grip_msg = self.generate_ee_action_msg(action)
                self.pub_robot_pose.publish(ee_msg)
                self.gripper_pub.publish(grip_msg)

        self.shared_obs["exec_done"] = len(self.pending_actions) == 0

def inference_loop(model_path, shared_obs, action_queue, action_horizon = 4, device = "cuda", start_time = 0, dt=0.1, action_exec_latency=0.2):

    model, cfg = load_policy(model_path)
    print("Inference process started.")
    
    # Load model configuration to get expected observation keys
    model_obs_keys = list(cfg.policy.shape_meta['obs'].keys())
    print(f"Model expects these observation keys: {model_obs_keys}")
    
    # Extract vision blackout configuration from the loaded policy
    blackout_config = {}
    try:
        dataset_cfg = cfg.task.dataset
        blackout_config = {
            'enabled': dataset_cfg.get('vision_blackout_enabled', False),
            'vision_keys': dataset_cfg.get('vision_blackout_keys', ['rs_wrist_rgb']),
            'tactile_threshold': dataset_cfg.get('vision_blackout_tactile_threshold', 0.1),
            'tactile_keys': dataset_cfg.get('vision_blackout_tactile_keys', ['dt_left_diff', 'dt_right_diff']),
        }

        if blackout_config['enabled']:
            print(f"🔌 Vision blackout ENABLED (tactile-based energy metric)")
            print(f"   Tactile threshold: {blackout_config['tactile_threshold']}")
            print(f"   Tactile keys: {blackout_config['tactile_keys']}")
            print(f"   Vision keys to blackout: {blackout_config['vision_keys']}")
        else:
            print("🔌 Vision blackout DISABLED")

    except AttributeError as e:
        print(f"Warning: Could not load vision blackout config: {e}")
        blackout_config['enabled'] = False

    # ─── Wait until first observation ───────────────────────────────
    while shared_obs.get("obs") is None:
        time.sleep(0.05)
        print("Waiting for first observation...")

    # Initialize previous timestamps based on available data
    prev_timestamps = {}
    obs_now = shared_obs["obs"]
    
    if "pose_timestamps" in obs_now:
        prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
    if "zed_rgb_timestamps" in obs_now:
        prev_timestamps["zed"] = obs_now["zed_rgb_timestamps"][-1]
    if "rs_side_rgb_timestamps" in obs_now:
        prev_timestamps["rs_side"] = obs_now["rs_side_rgb_timestamps"][-1]
    if "rs_front_rgb_timestamps" in obs_now:
        prev_timestamps["rs_front"] = obs_now["rs_front_rgb_timestamps"][-1]
    if "rs_wrist_rgb_timestamps" in obs_now:
        prev_timestamps["rs_wrist"] = obs_now["rs_wrist_rgb_timestamps"][-1]
    if "dt_left_timestamps" in obs_now:
        prev_timestamps["dt_left"] = obs_now["dt_left_timestamps"][-1]
    if "dt_right_timestamps" in obs_now:
        prev_timestamps["dt_right"] = obs_now["dt_right_timestamps"][-1]
    if "dt_left_diff_timestamps" in obs_now:
        prev_timestamps["dt_left_diff"] = obs_now["dt_left_diff_timestamps"][-1]
    if "dt_right_diff_timestamps" in obs_now:
        prev_timestamps["dt_right_diff"] = obs_now["dt_right_diff_timestamps"][-1]

    while True:
        loop_start_time = time.time()
        
        # If user paused with the keyboard
        if shared_obs.get("paused", False):
            time.sleep(0.05)
            continue

        wait_start_time = time.time()
        while True:
            obs_now = shared_obs["obs"]

            # Check if we have new data for all available sensors
            all_new = True
            waiting_on = []  # Track which streams we're waiting on
            
            if "pose_timestamps" in obs_now:
                pose_new = np.min(obs_now["pose_timestamps"]) > prev_timestamps.get("pose", -1)
                if not pose_new:
                    waiting_on.append("pose")
                all_new = all_new and pose_new
                
            if "zed_rgb_timestamps" in obs_now:
                zed_new = np.min(obs_now["zed_rgb_timestamps"]) > prev_timestamps.get("zed", -1)
                if not zed_new:
                    waiting_on.append("zed_rgb")
                all_new = all_new and zed_new
                
            if "rs_side_rgb_timestamps" in obs_now:
                rs_side_new = np.min(obs_now["rs_side_rgb_timestamps"]) > prev_timestamps.get("rs_side", -1)
                if not rs_side_new:
                    waiting_on.append("rs_side_rgb")
                all_new = all_new and rs_side_new
                
            if "rs_front_rgb_timestamps" in obs_now:
                rs_front_new = np.min(obs_now["rs_front_rgb_timestamps"]) > prev_timestamps.get("rs_front", -1)
                if not rs_front_new:
                    waiting_on.append("rs_front_rgb")
                all_new = all_new and rs_front_new
                
            if "rs_wrist_rgb_timestamps" in obs_now:
                rs_wrist_new = np.min(obs_now["rs_wrist_rgb_timestamps"]) > prev_timestamps.get("rs_wrist", -1)
                if not rs_wrist_new:
                    waiting_on.append("rs_wrist_rgb")
                all_new = all_new and rs_wrist_new

            if "dt_left_timestamps" in obs_now:
                dt_left_new = np.min(obs_now["dt_left_timestamps"]) > prev_timestamps.get("dt_left", -1)
                if not dt_left_new:
                    waiting_on.append("dt_left")
                all_new = all_new and dt_left_new

            if "dt_right_timestamps" in obs_now:
                dt_right_new = np.min(obs_now["dt_right_timestamps"]) > prev_timestamps.get("dt_right", -1)
                if not dt_right_new:
                    waiting_on.append("dt_right")
                all_new = all_new and dt_right_new

            if "dt_left_diff_timestamps" in obs_now:
                dt_left_diff_new = np.min(obs_now["dt_left_diff_timestamps"]) > prev_timestamps.get("dt_left_diff", -1)
                if not dt_left_diff_new:
                    waiting_on.append("dt_left_diff")
                all_new = all_new and dt_left_diff_new

            if "dt_right_diff_timestamps" in obs_now:
                dt_right_diff_new = np.min(obs_now["dt_right_diff_timestamps"]) > prev_timestamps.get("dt_right_diff", -1)
                if not dt_right_diff_new:
                    waiting_on.append("dt_right_diff")
                all_new = all_new and dt_right_diff_new

            if all_new:
                break
            
            # Check if we've been waiting too long and report what we're waiting on
            current_wait_time = time.time() - wait_start_time
            if current_wait_time > 0.008: 
                print(f"⚠️  Waiting {current_wait_time*1000:.1f}ms for new data from: {waiting_on}")
            
            time.sleep(0.001)
        
        wait_time = time.time() - wait_start_time

        # Save newest observation timestamps
        prep_start_time = time.time()
        if "pose_timestamps" in obs_now:
            prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
        if "zed_rgb_timestamps" in obs_now:
            prev_timestamps["zed"] = obs_now["zed_rgb_timestamps"][-1]
        if "rs_side_rgb_timestamps" in obs_now:
            prev_timestamps["rs_side"] = obs_now["rs_side_rgb_timestamps"][-1]
        if "rs_front_rgb_timestamps" in obs_now:
            prev_timestamps["rs_front"] = obs_now["rs_front_rgb_timestamps"][-1]
        if "rs_wrist_rgb_timestamps" in obs_now:
            prev_timestamps["rs_wrist"] = obs_now["rs_wrist_rgb_timestamps"][-1]
        if "dt_left_timestamps" in obs_now:
            prev_timestamps["dt_left"] = obs_now["dt_left_timestamps"][-1]
        if "dt_right_timestamps" in obs_now:
            prev_timestamps["dt_right"] = obs_now["dt_right_timestamps"][-1]
        if "dt_left_diff_timestamps" in obs_now:
            prev_timestamps["dt_left_diff"] = obs_now["dt_left_diff_timestamps"][-1]
        if "dt_right_diff_timestamps" in obs_now:
            prev_timestamps["dt_right_diff"] = obs_now["dt_right_diff_timestamps"][-1]

        # Grab new observations
        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_now.items()}

        # Extract policy observations - only include keys that the model expects
        model_obs = {}
        for k in model_obs_keys:
            if k in obs:
                model_obs[k] = obs[k]
            else:
                print(f"Warning: Model expects '{k}' but it's not available in observations")
        
        # Apply vision blackout based on policy configuration
        model_obs = apply_vision_blackout(model_obs, blackout_config)
        
        print(f"Sending to model: {list(model_obs.keys())}")
        
        prep_time = time.time() - prep_start_time

        # Save the most recent observation for debugging
        save_data(obs, debug_dir=DEBUG_DIR)

        # Predict an action-horizon batch
        inference_start_time = time.time()
        with torch.no_grad():
            actions = model.predict_action(model_obs)["action"][0].detach().cpu().numpy()
        inference_time = time.time() - inference_start_time

        print(f"Actions size: {actions.shape}")
        q_actions = actions[0:action_horizon]

        # ─── Schedule actions with timestamps ───────────────────────────────
        queue_start_time = time.time()
        
        # Record the start time of this inference cycle
        t_start = time.monotonic()
        
        # Compute scheduled timestamps for each predicted action
        action_timestamps = np.array([t_start + (i+1) * dt for i in range(len(q_actions))])
        current_time = time.monotonic()

       
        action_queue.put(("CLEAR_PENDING", current_time))
        print("🗑️  Sent CLEAR_PENDING signal to remove outdated actions")
        
        # Filter out any actions that would be executed too soon (add latency margin)
        valid_idx = action_timestamps > (current_time + action_exec_latency)
        if np.sum(valid_idx) == 0:
            # If no actions remain valid, use the last action and schedule it for the next time slot
            next_step_idx = int(np.ceil((current_time - t_start) / dt))
            action_timestamps = np.array([t_start + next_step_idx * dt])
            q_actions = q_actions[-1:]
            print("⚠️  All actions scheduled too soon, using last action with next available slot")
        else:
            q_actions = q_actions[valid_idx]
            # Reschedule valid actions starting from current time + latency
            action_start_time = current_time
            action_timestamps = np.array([action_start_time + i * dt for i in range(len(q_actions))])
            print(f"✅ Rescheduling {len(q_actions)} valid actions starting from now)")
        
        # Queue actions with their scheduled execution timestamps
        for act, ts in zip(q_actions, action_timestamps):
            action_queue.put((act, ts))
            
        queue_time = time.time() - queue_start_time

        time.sleep(.15)

        total_loop_time = time.time() - loop_start_time
        

def monitor_keys(policy_node):
    """
    Monitor key presses and handle pause, resume, and reset based on 'p', 'u', and 'r' keys.
    """
    print("Key controls:")
    print("  'p' - Pause policy execution")
    print("  'u' - Resume policy execution") 
    print("  'r' - Reset robot to initial position")
    print("  Close pygame window or Ctrl+C to exit")
    
    try:
        # Ensure pygame is properly initialized
        if not pygame.get_init():
            pygame.init()
        
        # Check if display is available
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((300, 200))
            pygame.display.set_caption("Robot Control - Use 'p', 'u', 'r' keys")
        
        while True:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # Handle window close event
                        print("Pygame window closed, exiting...")
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            print("Key 'p' pressed - Pausing policy")
                            policy_node.pause_policy()
                        elif event.key == pygame.K_u:
                            print("Key 'u' pressed - Resuming policy")
                            policy_node.resume_policy()
                        elif event.key == pygame.K_r:
                            print("Key 'r' pressed - Resetting robot")
                            policy_node.reset_xarm()

                time.sleep(0.01)
            except pygame.error as e:
                print(f"Pygame error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Error in monitor_keys: {e}")
        print("Keyboard monitoring disabled")

def main(args=None):
    """Main function to initialize the policy node and start the inference loop."""

    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    rclpy.init(args=args)

    use_pygame = True  # Enable pygame by default

    # Create shared memory structures
    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=False, exec_done=True)
    action_queue = Queue()
    start_time = time.monotonic()

    # Model path (relative to dt_ag root)
    dt_ag_root = pathlib.Path(__file__).resolve().parents[1]
    model_path = ""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scheduling parameters
    inference_action_horizon = 8
    dt = 0.05
    action_exec_latency = 0.25

    # Start inference process
    inference_process = Process(
        target=inference_loop, 
        args=(model_path, shared_obs, action_queue, inference_action_horizon, device, start_time, dt, action_exec_latency)
    )
    inference_process.daemon = True
    inference_process.start()

    # Create ROS node
    node = PolicyNode3D(shared_obs, action_queue, start_time, model_path)

    # Start pygame key monitoring thread
    if use_pygame:
        key_thread = threading.Thread(target=monitor_keys, args=(node,), daemon=True)
        key_thread.start()
        print("Pygame key monitoring enabled. Focus on the pygame window and use keyboard controls.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        inference_process.terminate()


if __name__ == '__main__':
    main()