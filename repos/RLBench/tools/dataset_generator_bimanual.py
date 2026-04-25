#!/usr/bin/env python3

import sys
import os
import logging
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np
import imageio

from rlbench import ObservationConfig
from rlbench.observation_config import CameraConfig

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointVelocity
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from rlbench.backend.exceptions import BoundaryError, InvalidActionError, TaskEnvironmentError, WaypointError, DemoError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.task import BIMANUAL_TASKS_PATH


import rich_click as click
from rich.logging import RichHandler
from click_prompt import choice_option
from click_prompt import filepath_option


camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]


def save_demo(demo, example_path, variation, save_video=False, video_camera="front", video_fps=30):
    data_types = ["rgb", "depth", "point_cloud", "mask"]
    #full_camera_names = list(map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types)))

    # Collect video frames if needed
    video_frames = [] if save_video else None

    # Save image data first, and then None the image data, and pickle
    for i, obs in enumerate(demo):
        for camera_name in camera_names:
            for dtype in data_types:

                camera_full_name = f"{camera_name}_{dtype}"
                data_path = os.path.join(example_path, camera_full_name)
                # ..todo:: actually I prefer to abort if this one exists
                os.makedirs(data_path, exist_ok=True)

                data = obs.perception_data.get(camera_full_name, None)

                if data is not None:
                    # Collect video frames from specified camera
                    if save_video and camera_name == video_camera and dtype == 'rgb':
                        video_frames.append(data.copy())

                    if dtype == 'rgb':
                        data = Image.fromarray(data)
                    elif dtype == 'depth':
                        data = utils.float_array_to_rgb_image(data, scale_factor=DEPTH_SCALE)
                    elif dtype == 'point_cloud':
                        continue
                    elif dtype == 'mask':
                        data = Image.fromarray((data * 255).astype(np.uint8))
                    else:
                        raise Exception('Invalid data type')
                    logging.debug("saving %s", camera_full_name)
                    data.save(os.path.join(data_path, f"{dtype}_{i:04d}.png"))

        # ..why don't we put everything into a pickle file?
        obs.perception_data.clear()

    # Save video if enabled
    if save_video and video_frames:
        video_path = os.path.join(example_path, f"demo_{video_camera}.mp4")
        imageio.mimsave(video_path, video_frames, fps=video_fps)
        logging.info("Saved video to %s", video_path)

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def run_all_variations(task_name, headless, save_path, episodes_per_task, image_size, save_video=False, video_camera="front", video_fps=30, ttt=None):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    from rich.logging import RichHandler
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    logging.root.name = task_name

    logging.info("Collecting data for %s", task_name)

    # ===== Scheme统计 =====
    scheme_stats = {'right_grasper': [], 'left_grasper': [], 'unknown': []}

    tasks = [task_file_to_task_class(task_name, True)]

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    default_config_params = {"image_size": image_size, "depth_in_meters": False, "masks_as_one_channel": False}
    camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
    obs_config.camera_configs = camera_configs


    # ..record date with BimanualJointPosition
    robot_setup = 'dual_panda'
    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup=robot_setup,
        headless=headless,
        ttt_file=ttt)

    rlbench_env.launch()

    tasks_with_problems = ""

    for task in tasks:
        
        task_env = rlbench_env.get_task(task)
        possible_variations = task_env.variation_count()

        logging.info("Task has %s possible variations", possible_variations)

        variation_path = os.path.join(save_path, task_env.get_name(), VARIATIONS_ALL_FOLDER)
        os.makedirs(variation_path, exist_ok=True)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        os.makedirs(episodes_path, exist_ok=True)


        abort_variation = False
        # for ex_idx in range(episodes_per_task):
        for ex_idx in range(140, 150):  # 从episode100到episode130，共30个
            attempts = 20           # 真正错误的重试次数
            scheme_skips = 0        # DemoError（方案过滤）计数
            max_scheme_skips = 20   # 方案过滤的最大重试次数（自由选择scheme时此值无影响）
            while attempts > 0 and scheme_skips < max_scheme_skips:
                try:
                    variation = np.random.randint(possible_variations)

                    task_env = rlbench_env.get_task(task)

                    task_env.set_variation(variation)

                    descriptions, obs = task_env.reset()

                    logging.info("// Task: %s Variation %s Demo %s", task_env.get_name(), variation, ex_idx)

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(amount=1, live_demos=True)

                    # ===== 获取并记录scheme信息 =====
                    active_scheme = 'unknown'
                    if hasattr(task_env._task, 'get_active_scheme'):
                        active_scheme = task_env._task.get_active_scheme()
                    elif hasattr(task_env._task, 'active_waypoint_mode'):
                        active_scheme = task_env._task.active_waypoint_mode

                except DemoError as e:
                    # DemoError 是方案过滤，不计入真正错误，单独计数
                    scheme_skips += 1
                    if scheme_skips % 10 == 0:
                        logging.info(f"Scheme skip #{scheme_skips}: {e}")
                    continue

                #  NoWaypointsError
                except (BoundaryError, WaypointError, InvalidActionError, TaskEnvironmentError) as e:
                    logging.warning("Exception %s", e)
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            task_env.get_name(), variation, ex_idx,str(e))
                    )
                    logging.error(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break

                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)

                save_demo(demo, episode_path, variation, save_video, video_camera, video_fps)

                with open(os.path.join( episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                    pickle.dump(descriptions, f)

                # ===== 记录scheme统计 =====
                if active_scheme in scheme_stats:
                    scheme_stats[active_scheme].append(ex_idx)
                else:
                    scheme_stats['unknown'].append(ex_idx)
                logging.info(f"Episode {ex_idx}: scheme={active_scheme}")

                # 保存scheme信息到episode目录
                scheme_info = {
                    'active_scheme': active_scheme,
                    'role_assignment': task_env._task.get_role_assignment() if hasattr(task_env._task, 'get_role_assignment') else {}
                }
                with open(os.path.join(episode_path, f'scheme_info_{active_scheme}.pkl'), 'wb') as f:
                    pickle.dump(scheme_info, f)

                break

            # 检查while循环退出原因
            if scheme_skips >= max_scheme_skips:
                logging.warning(f"Episode {ex_idx}: Skipped after {scheme_skips} scheme rejects. "
                                f"Target scheme may be rarely feasible for this task.")

            if abort_variation:
                break

        # ===== 打印scheme统计汇总 =====
        logging.info("=" * 60)
        logging.info(f"SCHEME STATISTICS for {task_env.get_name()}")
        logging.info("=" * 60)
        total_episodes = sum(len(v) for v in scheme_stats.values())
        for scheme_name, episodes in scheme_stats.items():
            if episodes:
                pct = len(episodes) / total_episodes * 100 if total_episodes > 0 else 0
                logging.info(f"  {scheme_name}: {len(episodes)} episodes ({pct:.1f}%)")
                logging.info(f"    Episode IDs: {episodes}")
        logging.info(f"  Total: {total_episodes} episodes")
        logging.info("=" * 60)

        # 保存汇总统计到文件
        stats_path = os.path.join(variation_path, 'scheme_statistics.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(scheme_stats, f)
        logging.info(f"Scheme statistics saved to {stats_path}")


    rlbench_env.shutdown()

    return tasks_with_problems



def get_bimanual_tasks():
    tasks =  [t.replace('.py', '') for t in
    os.listdir(BIMANUAL_TASKS_PATH) if t != '__init__.py' and t.endswith('.py')]
    return sorted(tasks)


@click.command()
@filepath_option("--save_path", default="/tmp/rlbench_data/",  help="Where to save the demos.")
@choice_option('--tasks', type=click.Choice(get_bimanual_tasks()), multiple=True, help='The tasks to collect. If empty, all tasks are collected.')
@click.option("--episodes_per_task", default=10, help="The number of episodes to collect per task.", prompt="Number of episodes")
@click.option("--all_variations", is_flag=True, default=True, help="Include all variations when sampling epsiodes")
#@click.option("--variations", default=-1, help="Number of variations to collect per task. -1 for all.")
@click.option("--headless/--no-headless", default=True, is_flag=True, help='Hide the simulator window')
#@click.option("--color-robot/--no-color-robot", default=False, is_flag=True, help='Colorize')
@choice_option('--image-size', type=click.Choice(["128x128", "256x256", "640x480"]), multiple=False, help='Select the image_size (width, height)')
@click.option("--save-video/--no-save-video", default=False, is_flag=True, help='Save demo video for each episode')
@click.option("--video-camera", default="front", type=click.Choice(camera_names), help='Camera to use for video recording')
@click.option("--video-fps", default=30, type=int, help='Video frame rate')
@click.option("--ttt", default=None, type=str, help='Custom TTT file (e.g., left_task_design_bimanual.ttt)')
def main(save_path, tasks, episodes_per_task, all_variations, headless, image_size, save_video, video_camera, video_fps, ttt):

    # ..todo check if already exits

    mp.set_start_method("spawn")

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    np.random.seed(None)

    ctx = mp.get_context('spawn')

    if not tasks:
        logging.error("No tasks selected!")


    logging.info("Generating %s episodes for each tasks %s with image size %s", episodes_per_task, tasks, image_size)


    image_size = list(map(int, image_size.split("x")))

    os.makedirs(save_path, exist_ok=True)

    if not all_variations:
        logging.error("Variations not supported")
        sys.exit(-1)

    logging.debug("Selected tasks %s", tasks)

    fn = partial(run_all_variations, headless=headless, save_path=save_path, episodes_per_task=episodes_per_task, image_size=image_size, save_video=save_video, video_camera=video_camera, video_fps=video_fps, ttt=ttt)
    with ctx.Pool(processes=8) as pool:
        pool.map(fn, tasks)


if __name__ == '__main__':
  main()


