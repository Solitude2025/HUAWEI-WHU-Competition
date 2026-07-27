REM 跌倒检测项目 - 一键流程（Windows）
REM 注意：请先在虚拟环境中运行，或用 venv 的 python 全路径替换 python

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

REM 1. 数据准备（下载 OmniFall -> 提取关键点 -> 分流）
REM    AV1 视频由 imageio-ffmpeg 解码；默认分层抽样 500 个视频，可用 --max_videos 调整
python prepare_data.py --all --max_videos 2000

REM 2. 训练（50 epochs，best 存 checkpoints/best.pth）
python train.py --data_dir data/train --val_dir data/val --epochs 50 --batch_size 32 --device cpu

REM 3. 阈值调优（在验证集上扫描最佳工作点）
python tune_threshold.py --checkpoint checkpoints/best.pth

REM 4. 评估报告 + 导出 ONNX
python evaluate.py --checkpoint checkpoints/best.pth
python export.py --checkpoint checkpoints/best.pth

REM 5. 推理（test/ 批量，输出标注视频到 outputs/）
python infer.py --test --checkpoint checkpoints/best.pth
