@echo off
REM ============================================
REM  跌倒检测一键运行脚本
REM  训练 → 测试集评估 → 推理
REM ============================================

echo.
echo [1/3] 训练模型 (50 epochs)...
python train.py --data_dir data/train --val_dir data/val --epochs 50 --batch_size 16

echo.
echo [2/3] 测试集评估...
python train.py --data_dir data/train --val_dir data/val --epochs 1 --batch_size 16

echo.
echo [3/3] 推理测试...
REM 请先把待推理视频放到 inference_test/ 目录
python infer.py --test --checkpoint checkpoints/best.pth

echo.
echo 完成! 结果保存在 outputs/ 目录