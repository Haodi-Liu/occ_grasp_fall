#!/usr/bin/env python3
"""
测试DINO特征提取功能

这个脚本测试:
1. DINOv2模型是否正确加载
2. 特征提取是否正常工作
3. 性能基准测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)


def test_dinov2_model_loading():
    """测试1: DINOv2模型加载"""
    print("\n" + "="*60)
    print("测试1: DINOv2模型加载")
    print("="*60)

    try:
        from agents.diffuser_actor_ppi.models.semantic_feature_extractor import Fusion

        print("初始化Fusion类...")
        fusion = Fusion(
            num_cam=3,
            feat_backbone='dinov2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print("✅ DINOv2模型加载成功!")
        print(f"   设备: {fusion.device}")
        print(f"   Backbone: {fusion.feat_backbone}")
        return fusion

    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        print("\n请先运行: bash scripts/download_dinov2_model.sh")
        return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dino_feature_extraction(fusion):
    """测试2: DINO特征提取"""
    print("\n" + "="*60)
    print("测试2: DINO特征提取")
    print("="*60)

    if fusion is None:
        print("⏭️  跳过 (模型未加载)")
        return False

    try:
        # 创建模拟数据
        K = 3  # 3个相机
        H, W = 128, 128
        N_points = 1000

        print(f"创建模拟数据:")
        print(f"  相机数: {K}")
        print(f"  图像大小: {H}x{W}")
        print(f"  点云数量: {N_points}")

        # RGB images (K, H, W, 3)
        rgb_images = np.random.randint(0, 255, (K, H, W, 3), dtype=np.uint8)

        # Depth images (K, H, W)
        depth_images = np.random.rand(K, H, W).astype(np.float32)

        # Camera poses (K, 3, 4)
        camera_poses = np.random.randn(K, 3, 4).astype(np.float32)

        # Camera intrinsics (K, 3, 3)
        camera_intrinsics = np.eye(3)[None].repeat(K, axis=0).astype(np.float32)
        camera_intrinsics[:, 0, 0] = 500  # fx
        camera_intrinsics[:, 1, 1] = 500  # fy
        camera_intrinsics[:, 0, 2] = W / 2  # cx
        camera_intrinsics[:, 1, 2] = H / 2  # cy

        # Point cloud (N, 3)
        point_cloud = torch.randn(N_points, 3, device=fusion.device, dtype=fusion.dtype)

        # 准备obs_dict
        obs_dict = {
            'color': rgb_images,
            'depth': depth_images,
            'pose': camera_poses,
            'K': camera_intrinsics
        }

        print("\n提取DINO特征...")
        start_time = time.time()

        dino_features = fusion.extract_semantic_feature_from_ptc(point_cloud, obs_dict)

        elapsed = time.time() - start_time

        print(f"✅ 特征提取成功!")
        print(f"   输入点云: {point_cloud.shape}")
        print(f"   输出特征: {dino_features.shape}")
        print(f"   耗时: {elapsed:.3f}秒")
        print(f"   特征范围: [{dino_features.min():.3f}, {dino_features.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dino_extractor_wrapper():
    """测试3: DINOFeatureExtractor包装类"""
    print("\n" + "="*60)
    print("测试3: DINOFeatureExtractor包装类")
    print("="*60)

    try:
        from agents.diffuser_actor_ppi.dino_feature_extractor import DINOFeatureExtractor

        print("初始化DINOFeatureExtractor...")
        extractor = DINOFeatureExtractor(
            num_cameras=3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable=True
        )

        # 创建模拟数据
        K = 3
        H, W = 128, 128
        N_points = 500

        rgb_images = np.random.randint(0, 255, (K, H, W, 3), dtype=np.uint8)
        depth_images = np.random.rand(K, H, W).astype(np.float32)
        camera_extrinsics = np.random.randn(K, 3, 4).astype(np.float32)
        camera_intrinsics = np.eye(3)[None].repeat(K, axis=0).astype(np.float32)

        point_cloud = torch.randn(N_points, 3)

        print("提取特征...")
        features = extractor.extract_features(
            point_cloud=point_cloud,
            rgb_images=rgb_images,
            depth_images=depth_images,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics
        )

        print(f"✅ 包装类工作正常!")
        print(f"   特征形状: {features.shape}")

        return True

    except Exception as e:
        print(f"❌ 包装类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """测试4: 性能基准测试"""
    print("\n" + "="*60)
    print("测试4: 性能基准测试")
    print("="*60)

    try:
        from agents.diffuser_actor_ppi.dino_feature_extractor import DINOFeatureExtractor

        extractor = DINOFeatureExtractor(num_cameras=3, enable=True)

        # 不同点云大小
        point_counts = [100, 500, 1000, 5000]
        K = 3
        H, W = 128, 128

        print(f"\n{'点云数量':<10} {'耗时(秒)':<12} {'FPS':<10}")
        print("-" * 40)

        for N in point_counts:
            rgb_images = np.random.randint(0, 255, (K, H, W, 3), dtype=np.uint8)
            depth_images = np.random.rand(K, H, W).astype(np.float32)
            camera_extrinsics = np.random.randn(K, 3, 4).astype(np.float32)
            camera_intrinsics = np.eye(3)[None].repeat(K, axis=0).astype(np.float32)
            point_cloud = torch.randn(N, 3)

            # 预热
            _ = extractor.extract_features(
                point_cloud, rgb_images, depth_images,
                camera_extrinsics, camera_intrinsics
            )

            # 计时
            n_runs = 5
            start = time.time()
            for _ in range(n_runs):
                _ = extractor.extract_features(
                    point_cloud, rgb_images, depth_images,
                    camera_extrinsics, camera_intrinsics
                )
            elapsed = (time.time() - start) / n_runs
            fps = 1.0 / elapsed

            print(f"{N:<10} {elapsed:<12.4f} {fps:<10.2f}")

        return True

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DINO特征提取功能测试")
    print("="*60)

    results = {}

    # 测试1: 模型加载
    fusion = test_dinov2_model_loading()
    results['model_loading'] = fusion is not None

    # 测试2: 特征提取
    results['feature_extraction'] = test_dino_feature_extraction(fusion)

    # 测试3: 包装类
    results['wrapper_class'] = test_dino_extractor_wrapper()

    # 测试4: 性能测试
    results['performance'] = test_performance_benchmark()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:<25} {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过!")
        print("\n下一步:")
        print("  1. 修改 diffuser_actor_agent.py 集成DINO提取")
        print("  2. 在配置文件中启用: enable_dino_features: true")
        print("  3. 运行训练测试")
    else:
        print("⚠️  部分测试失败，请检查:")
        print("  1. 是否运行了 bash scripts/download_dinov2_model.sh")
        print("  2. 检查错误信息并修复")
    print("="*60 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
