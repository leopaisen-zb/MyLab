#!/bin/bash

# 监控训练进度的脚本

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== 增强版EquiformerV2训练监控 =====${NC}"

# 检查训练日志
LOG_FILE="logs/train_enhanced_equiformer_v2.log"

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}等待训练日志文件创建...${NC}"
    while [ ! -f "$LOG_FILE" ]; do
        sleep 2
    done
fi

echo -e "${GREEN}发现训练日志文件，开始监控训练进度...${NC}"
echo ""

# 持续监控日志文件
tail -f "$LOG_FILE" | grep --line-buffered -E "Epoch|Training|Using device|Train Loss|Val Loss|Val MAE|Test"
