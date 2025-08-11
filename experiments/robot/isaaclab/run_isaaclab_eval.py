"""
run_isaaclab_eval.py

Evaluates a trained policy in an IsaacLab simulation task.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
import gymnasium as gym

# IsaacLab imports
from isaaclab.app import AppLauncher

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.isaaclab.isaaclab_utils import (
    get_isaac_dummy_action,
    get_isaac_image,
    get_isaac_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    normalize_gripper_action,
    invert_gripper_action,
    set_seed_everywhere,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


ASSET_BASE_PATH = Path("/data/ceph_hdd/main/artifactory/isaac_robocasa_assets/robocasa/new/robocasa/models/assets")
os.environ["ROBOCASA_ASSETS_ROOT"] = str(ASSET_BASE_PATH)
ASSET_PATH = Path("/home/zimu.gong/assets")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters (same as LIBERO)
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # IsaacLab environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"               # Task suite
    num_steps_wait: int = 2                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    device: str = "cuda:0"                           # Device for simulation and model

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # Adjust if different in IsaacLab
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    # TODO: adjust for IsaacLab tasks
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-IsaacLab-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_isaac_image(obs)
    wrist_img = get_isaac_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    pos = obs["policy"]["robot_ee_pose"][0][:3].cpu().numpy()
    quat_xyzw = obs["policy"]["robot_ee_pose"][0][[4, 5, 6, 3]].cpu().numpy()
    axisangle = quat2axisangle(quat_xyzw)
    gripper = obs["policy"]["joint_pos"][0][-2:].cpu().numpy()
    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            [pos,axisangle, gripper]
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    states, infos = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = 300  # Adjust based on task

    # Run episode
    success = False
    done = False
    # while t < max_steps * 100000 and not done:
    #     dummy_action = torch.zeros((1, 12))
    #     next_states, rewards, terminated, truncated, infos = env.step(dummy_action)
    #     done = terminated or truncated
    #     t += 1
    # try:
    x = 0.005
    while t < max_steps + cfg.num_steps_wait and not done:
        # Do nothing for the first few timesteps to let objects stabilize
        
        if t < cfg.num_steps_wait:
            states, infos = env.reset()
            next_states, rewards, terminated, truncated, infos = env.step(get_isaac_dummy_action(cfg.model_family))
            states = next_states
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(states, resize_size)
        replay_images.append(img)

        # If action queue is empty, requery model
        if len(action_queue) == 0:
            # Query model to get action
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(actions)

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, cfg.model_family)
        # Execute action in environment
        action = np.concatenate([action, np.zeros(11-len(action))]).reshape(1,-1)
        # action = np.concatenate([action[:-1], np.zeros(12-len(action[:-1]))]).reshape(1,-1)
        action = torch.tensor(action,dtype=torch.float32).to(cfg.device)
        # action = torch.tensor([[0.3, 0.0, 1.0+(t-10)*x, 
        #                     0.0, 1.0, 0.0, 0.0, 
        #                     1.0, 
        #                     0.0, 0.0, 0.0, 0.0]], device='cuda:0')
        
        
        next_states, rewards, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        if "success" in infos:  # Some tasks provide success in info
            success = infos["success"]
        elif done:
            states, infos = env.reset()
            success = True  # Assume done means success; adjust per task
        else:
            states = next_states
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_name: str,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""

    # robot_name = "LeRobot-RL"
    robot_name = "PandaOmron-Rel"
    # robot_name = "Panda-RL"
    scene_name = "robocasakitchen-1-8"
    robot_scale = 1.0
    num_envs = 1

    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    from isaaclab_tasks.utils import import_packages
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages("tasks", _BLACKLIST_PKGS)

    # Parse env config
    env_cfg = parse_env_cfg(
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        robot_scale=robot_scale,
        asset_base_path=ASSET_BASE_PATH,
        device=cfg.device,
        num_envs=num_envs,
        use_fabric=True,
        first_person_view=False,
        enable_cameras=app_launcher._enable_cameras,
        execute_mode=ExecuteMode.TRAIN
    )
    task_name = f"Robocasa-{task_name}-{robot_name}-v0"

    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={},
        disable_env_checker=True,
    )

    # Create environment
    env = gym.make(task_name, cfg=env_cfg)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video (reuse from LIBERO)
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Close env
    env.close()

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    return total_episodes, total_successes


@draccus.wrap()
def eval_isaaclab(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on IsaacLab tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Define example task list (add more tasks as needed; ensure they are vision-based and compatible)
    task_list = [
        # {"name": "OpenDrawerrl", "description": "open drawer"},
        {"name": "LiftObj", "description": "Pick up the object on the table"},
    ]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task in task_list:
        total_episodes, total_successes = run_task(
            cfg,
            task["name"],
            task["description"],
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    # Launch the app (required for IsaacLab)
    app_launcher = AppLauncher(dict(enable_cameras=True, headless=False))  # Adjust headless as needed
    simulation_app = app_launcher.app

    from lwlab.utils.env import parse_env_cfg, ExecuteMode

    eval_isaaclab()
