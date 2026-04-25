#!/usr/bin/env python3
"""
双臂角色分配统计分析脚本

分析收集的演示数据中的角色分配情况（哪只手抓取，哪只手推动）。
通过分析gripper状态变化来推断角色：
- 抓取臂 (grasper): gripper从打开(1.0)变为闭合(0.0)
- 推动臂 (pusher): gripper保持打开状态

用法:
    python analyze_role_assignment.py --data_path /path/to/task.train
    python analyze_role_assignment.py --data_path /mnt/rlbench_data/bimanual_edge_phone.train --verbose
"""

import argparse
import pickle
import os
import numpy as np
from collections import Counter


def analyze_episode(ep_path: str) -> dict:
    """分析单个episode的角色分配"""
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), 'rb') as f:
        demo = pickle.load(f)

    # 分析gripper状态变化
    left_start = demo[0].left.gripper_open
    left_end = demo[-1].left.gripper_open
    right_start = demo[0].right.gripper_open
    right_end = demo[-1].right.gripper_open

    # 判断哪只手进行了抓取（从打开变成闭合）
    left_grasped = left_start > 0.5 and left_end < 0.5
    right_grasped = right_start > 0.5 and right_end < 0.5

    if right_grasped and not left_grasped:
        role = "right_grasp_left_push"
        grasper = "right"
    elif left_grasped and not right_grasped:
        role = "left_grasp_right_push"
        grasper = "left"
    elif left_grasped and right_grasped:
        role = "both_grasped"
        grasper = "both"
    else:
        role = "neither_grasped"
        grasper = "none"

    # 获取物体位置信息（如果有）
    grasp_pos = demo[0].misc.get('grasp_position', None)
    contact_pos = demo[0].misc.get('contact_position', None)

    return {
        'role': role,
        'grasper': grasper,
        'left_gripper': (left_start, left_end),
        'right_gripper': (right_start, right_end),
        'grasp_position': grasp_pos,
        'contact_position': contact_pos,
        'demo_length': len(demo),
    }


def main():
    parser = argparse.ArgumentParser(description='分析双臂演示数据的角色分配')
    parser.add_argument('--data_path', type=str, required=True,
                        help='演示数据路径，例如 /mnt/rlbench_data/bimanual_edge_phone.train')
    parser.add_argument('--verbose', action='store_true',
                        help='显示每个episode的详细信息')
    args = parser.parse_args()

    # 查找episodes目录
    episodes_dir = os.path.join(args.data_path, "all_variations/episodes")
    if not os.path.exists(episodes_dir):
        print(f"错误: 找不到episodes目录: {episodes_dir}")
        return

    # 统计结果
    role_stats = Counter()
    episodes_info = []
    grasp_y_coords = {'right': [], 'left': [], 'both': [], 'none': []}

    episode_names = sorted(os.listdir(episodes_dir))
    print(f"\n正在分析 {len(episode_names)} 个episodes...\n")

    for ep_name in episode_names:
        ep_path = os.path.join(episodes_dir, ep_name)
        if not os.path.isdir(ep_path):
            continue

        try:
            info = analyze_episode(ep_path)
            info['episode'] = ep_name
            role_stats[info['role']] += 1
            episodes_info.append(info)

            # 记录Y坐标用于分析
            if info['grasp_position'] is not None:
                grasp_y_coords[info['grasper']].append(info['grasp_position'][1])

        except Exception as e:
            print(f"警告: 无法分析 {ep_name}: {e}")

    # 打印统计结果
    print("=" * 70)
    print("双臂角色分配统计报告")
    print("=" * 70)
    print(f"\n数据路径: {args.data_path}")
    print(f"总Episode数: {len(episodes_info)}")

    print("\n" + "-" * 70)
    print("角色分配统计:")
    print("-" * 70)
    for role, count in role_stats.most_common():
        pct = count / len(episodes_info) * 100
        description = {
            'right_grasp_left_push': '右臂抓取, 左臂推动',
            'left_grasp_right_push': '左臂抓取, 右臂推动',
            'both_grasped': '两臂都抓取 (异常)',
            'neither_grasped': '没有抓取动作 (异常)',
        }.get(role, role)
        print(f"  {description}: {count} ({pct:.1f}%)")

    # 分析物体位置与角色的关系
    if any(grasp_y_coords[k] for k in grasp_y_coords):
        print("\n" + "-" * 70)
        print("物体Y坐标分析 (Y<0偏右臂侧, Y>0偏左臂侧):")
        print("-" * 70)

        all_y = []
        for grasper, coords in grasp_y_coords.items():
            if coords:
                all_y.extend(coords)
                mean_y = np.mean(coords)
                print(f"  grasper={grasper}: 平均Y={mean_y:.4f}, 范围=[{min(coords):.4f}, {max(coords):.4f}]")

        if all_y:
            right_side = sum(1 for y in all_y if y < 0)
            left_side = len(all_y) - right_side
            print(f"\n  物体位置分布: 偏右侧(Y<0)={right_side}个, 偏左侧(Y>=0)={left_side}个")

    # Verbose模式下显示详细信息
    if args.verbose:
        print("\n" + "=" * 70)
        print("详细信息:")
        print("=" * 70)
        for info in episodes_info[:20]:  # 只显示前20个
            y_str = f"Y={info['grasp_position'][1]:.4f}" if info['grasp_position'] is not None else "N/A"
            print(f"{info['episode']}: grasper={info['grasper']}, "
                  f"left: {info['left_gripper'][0]:.1f}->{info['left_gripper'][1]:.1f}, "
                  f"right: {info['right_gripper'][0]:.1f}->{info['right_gripper'][1]:.1f}, "
                  f"grasp_pos {y_str}")

        if len(episodes_info) > 20:
            print(f"  ... 还有 {len(episodes_info) - 20} 个episodes")

    # 给出建议
    print("\n" + "=" * 70)
    print("分析结论:")
    print("=" * 70)

    if len(role_stats) == 1:
        dominant_role = list(role_stats.keys())[0]
        if dominant_role == 'right_grasp_left_push':
            print("  所有数据都是右臂抓取。角色选择算法可能未生效或需要调整。")
            print("  建议检查 ArmRoleSelector 的成本计算逻辑，或增加物体位置的随机化范围。")
        elif dominant_role == 'left_grasp_right_push':
            print("  所有数据都是左臂抓取。角色选择算法可能未生效或需要调整。")
    else:
        print(f"  数据包含多种角色分配，分布较为合理。")


if __name__ == '__main__':
    main()
