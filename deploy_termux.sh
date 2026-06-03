#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# Phigros 难度预测 - Termux 手机部署脚本
# 用法: bash deploy_termux.sh
# ============================================================

set -e

echo "=============================="
echo " Phigros 难度预测 - 手机部署"
echo "=============================="

# 1. 更新 Termux 包
echo "[1/5] 更新包管理器..."
pkg update -y && pkg upgrade -y

# 2. 安装 Python 和依赖
echo "[2/5] 安装 Python 和编译工具..."
pkg install -y python python-pip clang binutils libopenblas liblapack

# 3. 安装 Python 包
echo "[3/5] 安装 Python 依赖..."
pip install --upgrade pip
pip install Flask numpy scikit-learn

# 4. 下载项目文件（方式A: 从 GitHub）
echo "[4/5] 下载项目文件..."
echo ""
echo "请选择部署方式:"
echo "  A) 从 GitHub 克隆 (推荐，需先上传到 GitHub)"
echo "  B) 手动复制文件 (用 USB 或 SendAnywhere)"
echo ""
read -p "输入 A 或 B: " METHOD

if [ "$METHOD" = "A" ] || [ "$METHOD" = "a" ]; then
    read -p "输入 GitHub 仓库地址: " GIT_URL
    cd ~
    git clone "$GIT_URL" phigros_difficulty_estimator
    cd phigros_difficulty_estimator
else
    echo "请手动创建 ~/phigros_difficulty_estimator/ 目录"
    echo "并将以下文件复制进去:"
    echo "  - app.py"
    echo "  - feature_extractor.py"
    echo "  - predict_rpe.py"
    echo "  - unified_parser.py"
    echo "  - data_loader.py"
    echo "  - requirements.txt"
    echo "  - models/5dim_model_v5_2.pkl"
    echo "  - templates/index.html"
    echo ""
    read -p "复制完成后按 Enter 继续..."
    cd ~/phigros_difficulty_estimator
fi

# 5. 启动服务
echo "[5/5] 启动预测服务..."
echo ""
echo "=============================="
echo " 启动成功!"
echo " 手机内访问: http://127.0.0.1:5000"
echo " 电脑访问:   http://$(hostname -I 2>/dev/null | awk '{print $1}'):5000"
echo " 长按悬浮窗关闭服务"
echo "=============================="
echo ""

# 允许从外网访问（同一 WiFi）
export FLASK_APP=app.py
export FLASK_ENV=production
python app.py --host=0.0.0.0 --port=5000
