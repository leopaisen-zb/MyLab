#!/usr/bin/env bash
# scripts/retrain_equiformer.sh
#
# 飞轮⑧：Equiformer 预测模型一键重训脚本
#
# Demo 用法（答辩演示，无需 GPU，跳过真实训练直接注册 mock 版本）：
#   bash scripts/retrain_equiformer.sh --demo
#
# 真实训练（需要 GPU 环境 + CUDA + 正确 checkpoint 路径）：
#   bash scripts/retrain_equiformer.sh
#
# 可选参数：
#   --demo            跳过真实训练，注册 mock 版本
#   --db-path PATH    StateStore 数据库路径（默认：config.DB_PATH）
#   --registry-path   版本注册表 JSON 路径（默认：workspace/model_versions.json）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== MatGen-Eq Equiformer 重训脚本 ==="
echo "工作目录: $PROJECT_DIR"

# 切换到项目目录确保相对路径正确
cd "$PROJECT_DIR"

# 将脚本参数透传给 Python CLI
python -m backend.retrain --model-kind prediction "$@"

echo "=== 重训完成 ==="
