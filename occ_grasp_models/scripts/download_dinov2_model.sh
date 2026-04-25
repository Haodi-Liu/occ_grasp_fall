#!/bin/bash
# 下载DINOv2预训练模型

set -e

echo "==================================="
echo "下载DINOv2预训练模型"
echo "==================================="

# 创建目录
BASE_DIR="/home/hdliu/occ_grasp_fall"
MODELS_DIR="${BASE_DIR}/pretrained_models/hub/checkpoints"
REPO_DIR="${BASE_DIR}/repos"

mkdir -p "${MODELS_DIR}"
mkdir -p "${REPO_DIR}"

# 1. 克隆DINOv2仓库 (如果不存在)
if [ ! -d "${REPO_DIR}/dinov2" ]; then
    echo "克隆DINOv2仓库..."
    cd "${REPO_DIR}"
    git clone https://github.com/facebookresearch/dinov2.git
    echo "✅ DINOv2仓库克隆完成"
else
    echo "✅ DINOv2仓库已存在"
fi

# 2. 下载DINOv2 ViT-S/14预训练权重
MODEL_FILE="${MODELS_DIR}/dinov2_vits14_pretrain.pth"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "下载DINOv2 ViT-S/14预训练权重..."
    cd "${MODELS_DIR}"

    # Facebook官方链接
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
        -O dinov2_vits14_pretrain.pth

    if [ $? -eq 0 ]; then
        echo "✅ 模型下载完成: ${MODEL_FILE}"
        ls -lh "${MODEL_FILE}"
    else
        echo "❌ 下载失败，尝试备用链接..."
        # 备用链接（如果有的话）
        exit 1
    fi
else
    echo "✅ 模型已存在: ${MODEL_FILE}"
    ls -lh "${MODEL_FILE}"
fi

echo ""
echo "==================================="
echo "安装完成! 文件结构:"
echo "==================================="
echo "DINOv2仓库: ${REPO_DIR}/dinov2"
echo "预训练权重: ${MODEL_FILE}"
echo ""
echo "使用方法:"
echo "  from agents.diffuser_actor_ppi.models.semantic_feature_extractor import Fusion"
echo "  fusion = Fusion(num_cam=3, feat_backbone='dinov2')"
echo ""
