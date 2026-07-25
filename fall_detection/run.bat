pip install -r requirements.txt
python prepare_data.py
python train.py --data_dir data/train --val_dir data/val --epochs 100
python infer.py --video test.mp4 --checkpoint checkpoints/best.pth --save