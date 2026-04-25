"""
Scheme Utilities for Bimanual Task Evaluation

This module provides utilities for building episode-to-scheme mappings
from demonstration data, enabling scheme-stratified evaluation metrics.

重要：本模块只处理【不带 .train 后缀】的评估数据目录。

Usage:
    from helpers.scheme_utils import build_episode_scheme_map, get_scheme_stats_summary

    # 构建 episode → scheme 映射
    episode_scheme_map = build_episode_scheme_map('/mnt/rlbench_data', 'bimanual_edge_phone')
    # 返回: {0: 'left_grasper', 1: 'right_grasper', 2: 'left_grasper', ...}

    # 计算汇总指标
    scheme_summary = get_scheme_stats_summary(scheme_stats)
    # 返回: {'success_rate_left_grasper_scenes': 0.8, ...}
"""

import os
import logging
from typing import Dict

try:
    from natsort import natsorted
except ImportError:
    # 如果 natsort 不可用，使用简单的自然排序替代
    import re
    def natsorted(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


def build_episode_scheme_map(dataset_root: str, task_name: str) -> Dict[int, str]:
    """
    构建 episode 编号到 scheme 的映射。

    通过扫描演示数据目录中的 scheme_info_*.pkl 文件名来获取每个 episode 的 GT scheme。

    Args:
        dataset_root: 数据集根目录
            示例: '/mnt/rlbench_data'
        task_name: 任务名称（不带 .train 后缀）
            示例: 'bimanual_edge_phone'

    Returns:
        Dict[int, str]: episode 编号到 scheme 的映射
            示例: {
                0: 'left_grasper',
                1: 'right_grasper',
                2: 'left_grasper',
                3: 'right_grasper',
                ...
            }

    Note:
        - 只扫描不带 .train 后缀的目录（评估数据）
        - 文件名格式: scheme_info_left_grasper.pkl 或 scheme_info_right_grasper.pkl
        - 如果某个 episode 没有 scheme_info 文件，该 episode 将被跳过
    """
    # 构建评估数据目录路径（不带 .train 后缀）
    # 示例: /mnt/rlbench_data/bimanual_edge_phone/all_variations/episodes/
    episodes_path = os.path.join(dataset_root, task_name, "all_variations", "episodes")

    if not os.path.exists(episodes_path):
        logging.warning(f"Episodes directory not found: {episodes_path}")
        return {}

    episode_scheme_map = {}

    try:
        # 获取所有 episode 目录，按自然顺序排序
        # 示例: ['episode0', 'episode1', 'episode10', 'episode11', ...]
        # natsorted 会正确排序为: ['episode0', 'episode1', ..., 'episode10', 'episode11', ...]
        episode_dirs = natsorted([
            d for d in os.listdir(episodes_path)
            if d.startswith('episode') and os.path.isdir(os.path.join(episodes_path, d))
        ])
    except Exception as e:
        logging.warning(f"Error listing episodes directory: {e}")
        return {}

    for ep_dir in episode_dirs:
        ep_path = os.path.join(episodes_path, ep_dir)

        # 提取 episode 编号
        # 'episode5' → 5
        try:
            ep_idx = int(ep_dir.replace('episode', ''))
        except ValueError:
            logging.warning(f"Cannot parse episode number from directory name: {ep_dir}")
            continue

        # 查找 scheme_info_*.pkl 文件
        try:
            for f in os.listdir(ep_path):
                if f.startswith('scheme_info_') and f.endswith('.pkl'):
                    # 'scheme_info_left_grasper.pkl' → 'left_grasper'
                    scheme = f.replace('scheme_info_', '').replace('.pkl', '')
                    episode_scheme_map[ep_idx] = scheme
                    break
        except Exception as e:
            logging.warning(f"Error reading episode {ep_idx}: {e}")
            continue

    if episode_scheme_map:
        # 统计 scheme 分布
        # 示例: {'left_grasper': 45, 'right_grasper': 35}
        scheme_counts = {}
        for scheme in episode_scheme_map.values():
            scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
        logging.info(f"Built scheme map for {task_name}: {len(episode_scheme_map)} episodes, "
                     f"distribution: {scheme_counts}")
    else:
        logging.warning(f"No scheme information found for task {task_name} at {episodes_path}")

    return episode_scheme_map


def get_scheme_stats_summary(scheme_stats: Dict[str, Dict[str, any]]) -> Dict[str, float]:
    """
    计算 scheme 分层统计的汇总指标。

    Args:
        scheme_stats: scheme 统计字典
            示例: {
                'left_grasper': {'success': 10, 'total': 15, 'success_steps': [120, 130, ...]},
                'right_grasper': {'success': 8, 'total': 12, 'success_steps': [110, 125, ...]},
                'unknown': {'success': 0, 'total': 3, 'success_steps': []}
            }

    Returns:
        Dict[str, float]: 汇总指标
            示例: {
                'success_rate_left_grasper_scenes': 0.667,    # 10/15
                'success_rate_right_grasper_scenes': 0.667,   # 8/12
                'scheme_balance_gap': 0.0,                    # |0.667 - 0.667|
                'total_left_grasper_episodes': 15,
                'total_right_grasper_episodes': 12,
                'avg_steps_left_grasper_scenes': 125.0,       # 成功案例平均步数
                'avg_steps_right_grasper_scenes': 117.5       # 成功案例平均步数
            }
    """
    summary = {}

    rates = {}
    for scheme in ['left_grasper', 'right_grasper']:
        stats = scheme_stats.get(scheme, {'success': 0, 'total': 0, 'success_steps': []})
        total = stats['total']
        success = stats['success']

        # 计算成功率
        rate = success / total if total > 0 else 0.0
        rates[scheme] = rate

        summary[f'success_rate_{scheme}_scenes'] = rate
        summary[f'total_{scheme}_episodes'] = total

        # 计算平均步数（仅成功案例）
        success_steps = stats.get('success_steps', [])
        if success_steps:
            avg_steps = sum(success_steps) / len(success_steps)
            summary[f'avg_steps_{scheme}_scenes'] = avg_steps

    # 计算 balance gap: 两种场景成功率的差距
    summary['scheme_balance_gap'] = abs(rates.get('left_grasper', 0) - rates.get('right_grasper', 0))

    return summary
