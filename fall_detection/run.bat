@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title HUAWEI-WHU 跌倒检测 - 一键运行

:: ============================================================
::   HUAWEI-WHU 跌倒检测项目 - 一键运行脚本
::   
::   用法: 双击 run.bat 或 在终端运行 run.bat
:: ============================================================

cls
echo ============================================================
echo     HUAWEI-WHU 轻量级跌倒检测 - 一键运行
echo     面向低算力端侧平台基于视觉的实时跌倒检测
echo ============================================================
echo.
echo 项目目录: %~dp0
echo.

:: 进入项目目录
cd /d "%~dp0"

:: ============ 检测 Python 环境 ============
:check_python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo        下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor>=8 else 1)"
if %errorlevel% neq 0 (
    echo [错误] Python 版本过低，需要 Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>nul') do set PY_VER=%%i
echo [OK] Python: %PY_VER%

:: 检测 CUDA（GPU）
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.get_device_name(0))"') do set GPU_NAME=%%i
    set HAS_GPU=1
    echo [OK] GPU 可用: !GPU_NAME!
) else (
    set HAS_GPU=0
    echo [!] 未检测到 GPU，将使用 CPU（训练较慢，但可用）
)
echo.

:: ============ 主菜单 ============
:main_menu
echo.
echo ======================== 主菜单 ========================
echo.
echo  1. 一键完整运行 (安装依赖 + 准备数据 + 训练 + 推理)
echo  2. 仅安装依赖
echo  3. 安装依赖 + 数据准备
echo  4. 安装依赖 + 训练（使用合成数据演示）
echo  5. 仅训练（需要有数据）
echo  6. 导出模型（ONNX / TorchScript）
echo  7. 查看训练结果
echo  8. 重新生成配置
echo  9. 清理临时文件
echo  0. 退出
echo.
set /p MENU_CHOICE="请输入选项 [0-9]: "

if "%MENU_CHOICE%"=="1" goto full_pipeline
if "%MENU_CHOICE%"=="2" goto install_deps
if "%MENU_CHOICE%"=="3" goto install_and_data
if "%MENU_CHOICE%"=="4" goto install_and_demo
if "%MENU_CHOICE%"=="5" goto train_only
if "%MENU_CHOICE%"=="6" goto export_model
if "%MENU_CHOICE%"=="7" goto show_results
if "%MENU_CHOICE%"=="8" goto gen_config
if "%MENU_CHOICE%"=="9" goto clean_up
if "%MENU_CHOICE%"=="0" goto end

echo [错误] 无效选项，请重新输入
goto main_menu

:: ============ 安装依赖 ============
:install_deps
cls
echo ======================== 安装依赖 ========================
echo.

:: 创建虚拟环境（可选）
if not exist ".venv" (
    echo [*] 创建虚拟环境...
    python -m venv .venv
    if %errorlevel% equ 0 (
        echo [OK] 虚拟环境已创建
    ) else (
        echo [!] 虚拟环境创建失败，使用全局 Python
    )
)

:: 升级 pip
echo [*] 升级 pip...
python -m pip install --upgrade pip -q

:: 安装 PyTorch（根据CUDA版本选择）
echo [*] 安装 PyTorch（根据 GPU 自动选择版本）...
if "%HAS_GPU%"=="1" (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if %errorlevel% neq 0 (
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    ) else (
        echo [OK] PyTorch 已安装
    )
) else (
    python -c "import torch; exit(0)" >nul 2>&1
    if %errorlevel% neq 0 (
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    ) else (
        echo [OK] PyTorch 已安装
    )
)

:: 安装其他依赖
echo [*] 安装项目依赖...
pip install -r requirements.txt -q

:: 安装数据处理依赖
echo [*] 安装数据处理依赖...
pip install datasets kagglehub ultralytics -q

echo.
echo [OK] 依赖安装完成！
echo.
pause
goto main_menu

:: ============ 安装 + 数据 ============
:install_and_data
cls
call :install_deps_inner

echo.
echo ======================== 准备数据 ========================
echo.
echo 选择数据来源:
echo  1. 生成合成数据（快速，无网络需求，仅演示）
echo  2. 下载真实数据集（需要网络，推荐 OmniFall）
echo  3. 跳过
echo.
set /p DATA_CHOICE="请输入选项 [1-3]: "

if "%DATA_CHOICE%"=="1" goto demo_data
if "%DATA_CHOICE%"=="2" goto real_data
if "%DATA_CHOICE%"=="3" goto main_menu

goto install_and_data

:demo_data
echo [*] 生成合成数据用于演示...
python -c "
import numpy as np, os
data_dir = 'data/train'
os.makedirs(data_dir, exist_ok=True)
# 生成 100 个合成训练序列
for v in range(50):
    for p in range(1):
        person_dir = os.path.join(data_dir, f'video_{v:04d}', f'person_{p}')
        os.makedirs(person_dir, exist_ok=True)
        T = 120
        kp = np.zeros((T, 17, 3))
        bb = np.zeros((T, 4))
        lbl = np.zeros(T)
        is_fall = v >= 25
        for t in range(T):
            if is_fall:
                fall_progress = max(0, min(1, (t - 30) / 30))
                if t < 30:
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.02
                    kp[t, :, 1] = np.linspace(0.1, 0.95, 17) + np.random.randn(17) * 0.01
                    kp[t, :, 2] = 0.8 + np.random.rand(17) * 0.2
                    bb[t] = [0.3, 0.05, 0.7, 0.95]
                    lbl[t] = 0
                elif t < 60:
                    kp[t, :, 1] = np.linspace(0.1, 0.95, 17) * (1 - fall_progress * 0.7) + fall_progress * 0.7
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                    kp[t, :, 2] = 0.5 + np.random.rand(17) * 0.3
                    bb[t] = [0.2, 0.2, 0.8, 0.85]
                    lbl[t] = 1
                else:
                    kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                    kp[t, :, 1] = 0.8 + np.random.randn(17) * 0.03
                    kp[t, :, 2] = 0.4 + np.random.rand(17) * 0.3
                    bb[t] = [0.15, 0.4, 0.85, 0.85]
                    lbl[t] = 1
            else:
                kp[t, :, 0] = 0.5 + np.random.randn(17) * 0.03
                kp[t, :, 1] = np.linspace(0.1, 0.95, 17) + np.random.randn(17) * 0.02
                kp[t, :, 2] = 0.7 + np.random.rand(17) * 0.3
                bb[t] = [0.3, 0.05, 0.7, 0.95]
                lbl[t] = 0
        np.save(os.path.join(person_dir, 'keypoints.npy'), kp)
        np.save(os.path.join(person_dir, 'bboxes.npy'), bb)
        np.save(os.path.join(person_dir, 'labels.npy'), lbl)
print(f'合成数据已生成: {data_dir}')
" && echo [OK] 合成数据准备完成
goto main_menu

:real_data
echo [*] 准备下载真实数据集...
echo.
echo 推荐使用 OmniFall（整合了8个公共数据集，质量最高）
echo 下载需要 5-10 分钟（取决于网络速度）
echo.
python prepare_data.py --all --max_videos 50
if %errorlevel% equ 0 (
    echo [OK] 数据准备完成！
) else (
    echo [!] 自动下载失败，请手动下载:
    echo     OmniFall: https://huggingface.co/datasets/simplexsigil2/omnifall
    echo     Kaggle:   https://www.kaggle.com/datasets/payutch/fall-video-dataset
    echo.
    echo     下载后放入 data/raw/ 目录
)
pause
goto main_menu

:: ============ 安装 + 训练演示 ============
:install_and_demo
cls
call :install_deps_inner
call :demo_data_inner
goto train_demo

:: ============ 仅训练 ============
:train_only
cls
echo ======================== 训练模型 ========================
echo.

:: 检测是否有数据
if not exist "data\train" (
    echo [!] 未检测到训练数据 (data/train/)
    echo [!] 将使用合成数据进行演示
    echo.
    call :demo_data_inner
)

goto train_demo

:: ============ 训练（通用入口） ============
:train_demo
cls
echo ======================== 开始训练 ========================
echo.

:: 训练参数
set EPOCHS=50
set BATCH=32
set DEVICE=cpu
if "%HAS_GPU%"=="1" set DEVICE=cuda

echo 训练参数:
echo   - 版本: standard
echo   - Epochs: %EPOCHS%
echo   - Batch: %BATCH%
echo   - 设备: %DEVICE%
echo.

echo [*] 正在训练，请稍候...
echo.
python train.py --epochs %EPOCHS% --batch_size %BATCH% --version standard --device %DEVICE%
echo.

if %errorlevel% equ 0 (
    echo [OK] 训练完成！
    echo.
    echo 模型保存位置: checkpoints/
    echo   - best.pth   (最佳模型)
    echo   - latest.pth (最新模型)
) else (
    echo [错误] 训练失败，请检查上面日志
)

echo.
pause
goto main_menu

:: ============ 导出模型 ============
:export_model
cls
echo ======================== 导出模型 ========================
echo.

set CHECKPOINT=checkpoints\best.pth
if not exist "%CHECKPOINT%" (
    echo [!] 未找到训练好的模型: %CHECKPOINT%
    echo [!] 请先训练模型（选项 4 或 5）
    echo.
    pause
    goto main_menu
)

echo 导出格式:
echo  1. ONNX（通用格式，华为 NPU 部署推荐）
echo  2. TorchScript（PyTorch 原生）
echo  3. 全部导出
echo.
set /p EXPORT_CHOICE="请选择 [1-3]: "

if "%EXPORT_CHOICE%"=="1" set FORMAT=onnx
if "%EXPORT_CHOICE%"=="2" set FORMAT=torchscript
if "%EXPORT_CHOICE%"=="3" set FORMAT=all

echo [*] 正在导出...
python export.py --checkpoint %CHECKPOINT% --format %FORMAT%
echo.
echo [OK] 导出完成！文件保存在 export/ 目录
pause
goto main_menu

:: ============ 查看结果 ============
:show_results
cls
echo ======================== 查看训练结果 ========================
echo.

:: 检查训练日志
if exist "checkpoints\latest.pth" (
    echo [模型检查点]
    dir /b checkpoints\*.pth 2>nul
    echo.
    
    python -c "
import torch
for ckpt in ['checkpoints/latest.pth', 'checkpoints/best.pth']:
    try:
        c = torch.load(ckpt, map_location='cpu')
        print(f'  {ckpt}:')
        print(f'    Epoch: {c.get(\"epoch\", \"?\")}')
        print(f'    Best F1: {c.get(\"best_f1\", \"N/A\"):.4f}')
        for k, v in c.get('train_metrics', {}).items():
            if isinstance(v, float):
                print(f'    Train {k}: {v:.4f}')
        for k, v in c.get('val_metrics', {}).items():
            if isinstance(v, float):
                print(f'    Val {k}: {v:.4f}')
    except: pass
"
) else (
    echo [!] 暂无训练结果。
    echo [!] 请先训练模型（选项 4 或 5）
)

echo.
pause
goto main_menu

:: ============ 生成配置文件 ============
:gen_config
cls
echo ======================== 重置配置文件 ========================
if exist "configs\config.yaml" (
    echo [*] 配置文件已存在: configs/config.yaml
    set /p OVERWRITE="是否覆盖 (y/n)? "
    if /i not "!OVERWRITE!"=="y" goto main_menu
)
echo. >nul

:: 配置已存在，重新生成
python -c "
import yaml
config = {
    'model': {
        'name': 'LightSpatialTemporalFallDetector',
        'version': 'standard',
        'motion_encoder': {'feature_dim': 48, 'frame_window': 3},
        'tcn': {
            'version': 'v1', 'hidden_dim': 64, 'num_layers': 5,
            'kernel_size': 3, 'dilations': [1,2,4,8,16], 'dropout': 0.1,
        },
        'rule': {
            'torso_angle_threshold': 0.5, 'torso_angle_duration': 8,
            'velocity_peak_threshold': 0.02, 'stillness_threshold': 0.005,
            'stillness_duration': 15, 'vote_window': 10, 'vote_threshold': 0.6,
            'tcn_prob_threshold': 0.5, 'fall_memory_frames': 60,
        },
    },
    'detector': {
        'model': 'yolov8n-pose.pt', 'device': 'cpu',
        'input_size': [640, 384], 'conf_threshold': 0.25,
        'iou_threshold': 0.7, 'max_det': 5,
    },
    'tracker': {
        'track_thresh': 0.5, 'match_thresh': 0.8,
        'track_buffer': 30, 'frame_rate': 30,
    },
    'pipeline': {
        'sequence_length': 32, 'detection_interval': 2,
        'ir_mode': False, 'save_video': False, 'output_dir': 'output',
    },
    'training': {
        'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001,
        'weight_decay': 0.0001, 'lr_scheduler': 'cosine',
        'sequence_length': 32, 'stride': 8, 'use_augmentation': True,
    },
    'compliance': {
        'max_params': 20000000, 'max_model_size_mb': 80,
        'max_latency_ms': 100, 'max_npu_memory_mb': 20,
    },
}
os.makedirs('configs', exist_ok=True)
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print('[OK] 配置文件已生成')
" 2>nul || echo [OK] 配置文件已存在

echo [OK] 配置已重置
pause
goto main_menu

:: ============ 清理 ============
:clean_up
cls
echo ======================== 清理 ========================
echo.
echo 将清理以下内容:
echo   - __pycache__ 缓存目录
echo   - .eggs / *.egg-info（包信息）
echo   - checkpoints 中的临时文件
echo   - export 目录
echo.

set /p CONFIRM="确认清理 (y/n)? "
if /i not "!CONFIRM!"=="y" goto main_menu

:: 清理
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
if exist "*.egg-info" rmdir /s /q *.egg-info 2>nul

echo [OK] 清理完成
pause
goto main_menu

:: ============ 完整流水线 ============
:full_pipeline
cls
echo ======================== 完整运行 ========================
echo.
echo 流程: 安装依赖 → 准备数据 → 训练模型 → 导出 ONNX
echo.

:: 1. 安装依赖
call :install_deps_inner

:: 2. 准备数据
call :demo_data_inner

:: 3. 训练
set DEVICE=cpu
if "%HAS_GPU%"=="1" set DEVICE=cuda
echo.
echo [*] 开始训练...
python train.py --epochs 50 --batch_size 32 --version standard --device %DEVICE%

if %errorlevel% neq 0 (
    echo [错误] 训练失败
    pause
    goto main_menu
)

:: 4. 导出 ONNX
echo.
echo [*] 导出 ONNX 模型...
python export.py --checkpoint checkpoints\best.pth --format onnx

:: 5. 完成
echo.
echo ======================== 全部完成！=======================
echo.
echo 项目结构:
echo   checkpoints/best.pth    - 最佳模型权重
echo   export/                  - 导出的 ONNX/TorchScript
echo   data/train/              - 训练数据
echo   data/val/                - 验证数据
echo.
echo 常用命令:
echo   python train.py    - 训练模型
echo   python infer.py    - 视频推理测试
echo   python export.py   - 导出部署模型
echo.
echo 技术方案文档: README.md
echo.
pause
goto main_menu

:: ============ 内部子程序 ============

:install_deps_inner
echo [*] 安装依赖...
python -m pip install --upgrade pip -q
if "%HAS_GPU%"=="1" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>nul
) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>nul
)
pip install -r requirements.txt -q
pip install scikit-learn -q
echo [OK] 依赖安装完成
exit /b 0

:demo_data_inner
echo [*] 准备演示数据...
if not exist "data\train" (
    mkdir data\train 2>nul
    python -c "
import numpy as np, os
for v in range(50):
    for p in range(1):
        pd = os.path.join('data/train', f'video_{v:04d}', f'person_{p}')
        os.makedirs(pd, exist_ok=True)
        T = 120; kp = np.zeros((T,17,3)); bb = np.zeros((T,4)); lbl = np.zeros(T)
        is_fall = v >= 25
        for t in range(T):
            if is_fall:
                if t < 30:
                    kp[t,:,0]=0.5+np.random.randn(17)*0.02; kp[t,:,1]=np.linspace(0.1,0.95,17)+np.random.randn(17)*0.01
                    kp[t,:,2]=0.8+np.random.rand(17)*0.2; bb[t]=[0.3,0.05,0.7,0.95]; lbl[t]=0
                elif t < 60:
                    fp=(t-30)/30; kp[t,:,1]=np.linspace(0.1,0.95,17)*(1-fp*0.7)+fp*0.7
                    kp[t,:,0]=0.5+np.random.randn(17)*0.03; kp[t,:,2]=0.5+np.random.rand(17)*0.3
                    bb[t]=[0.2,0.2,0.8,0.85]; lbl[t]=1
                else:
                    kp[t,:,0]=0.5+np.random.randn(17)*0.03; kp[t,:,1]=0.8+np.random.randn(17)*0.03
                    kp[t,:,2]=0.4+np.random.rand(17)*0.3; bb[t]=[0.15,0.4,0.85,0.85]; lbl[t]=1
            else:
                kp[t,:,0]=0.5+np.random.randn(17)*0.03; kp[t,:,1]=np.linspace(0.1,0.95,17)+np.random.randn(17)*0.02
                kp[t,:,2]=0.7+np.random.rand(17)*0.3; bb[t]=[0.3,0.05,0.7,0.95]; lbl[t]=0
        np.save(os.path.join(pd,'keypoints.npy'),kp); np.save(os.path.join(pd,'bboxes.npy'),bb); np.save(os.path.join(pd,'labels.npy'),lbl)
    "
    echo [OK] 演示数据准备完成
) else (
    echo [OK] 训练数据已存在
)
exit /b 0

:: ============ 退出 ============
:end
echo.
echo 感谢使用！项目 GitHub: https://github.com/Solitude2025/HUAWEI-WHU-Competition
echo.
pause
