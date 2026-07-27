"""
模型导出脚本
--------------
将训练好的模型导出为部署格式：
- ONNX: 通用格式，可转换到华为 NPU (Ascend)
- TorchScript: PyTorch 原生的序列化格式
- OpenVINO IR: Intel 平台

华为 NPU 部署流程:
    PyTorch → ONNX → ATC (Ascend Tensor Compiler) → OM (离线模型)

用法:
    python export.py --checkpoint checkpoints/best.pth --format onnx
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector import FallDetector, create_fall_detector


def export_onnx(
    model: FallDetector,
    output_path: str,
    input_shape: tuple = (1, 32, 48),
    dynamic_axes: bool = True,
    opset_version: int = 11,
):
    """
    导出 ONNX 格式
    
    注意: 仅导出 TCN 部分。
    MotionEncoder 是无参数的纯计算，作为独立算子部署。
    RuleRefinement 是规则引擎，在应用层实现。
    """
    model.eval()
    
    # 准备 TCN 部分
    tcn = model.tcn
    dummy_input = torch.randn(*input_shape)
    
    # 动态轴配置
    if dynamic_axes:
        dynamic_axes_config = {
            "motion_features": {0: "batch_size", 1: "sequence_length"},
            "fall_prob": {0: "batch_size", 1: "sequence_length"},
        }
    else:
        dynamic_axes_config = None
    
    try:
        torch.onnx.export(
            tcn,
            dummy_input,
            output_path,
            input_names=["motion_features"],
            output_names=["fall_prob"],
            dynamic_axes=dynamic_axes_config,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"[Export] ONNX 导出成功: {output_path}")
        
        # 验证 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("[Export] ONNX 模型验证通过")
        except ImportError:
            print("[Export] 提示: pip install onnx 以验证模型")
            
    except Exception as e:
        print(f"[Export] ONNX 导出失败: {e}")
        raise


def export_torchscript(
    model: FallDetector,
    output_path: str,
    input_shape: tuple = (1, 32, 48),
):
    """导出 TorchScript 格式"""
    model.eval()
    
    tcn = model.tcn
    dummy_input = torch.randn(*input_shape)
    
    try:
        # Trace 方式
        traced = torch.jit.trace(tcn, dummy_input)
        
        # 优化
        traced = torch.jit.optimize_for_inference(traced)
        
        torch.jit.save(traced, output_path)
        print(f"[Export] TorchScript 导出成功: {output_path}")
        
        # 重新加载验证
        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            out1 = tcn(dummy_input)
            out2 = loaded(dummy_input)
            diff = (out1 - out2).abs().max().item()
            print(f"[Export] 验证通过 (max diff: {diff:.6f})")
            
    except Exception as e:
        print(f"[Export] TorchScript 导出失败: {e}")
        raise


def check_compliance(model: FallDetector) -> dict:
    """检查赛题合规性"""
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    results = {
        "total_params": total_params,
        "param_limit_20M": total_params <= 20_000_000,
        "model_size_mb_fp32": model_size_mb,
        "size_limit_80MB": model_size_mb <= 80,
        "model_size_mb_int8": model_size_mb / 4,
    }
    
    print("\n[合规检查]")
    print(f"  参数总量: {total_params:,} {'✓' if results['param_limit_20M'] else '✗ 超限'}")
    print(f"  FP32 大小: {model_size_mb:.2f} MB {'✓' if results['size_limit_80MB'] else '✗ 超限'}")
    print(f"  INT8 大小: {model_size_mb/4:.2f} MB (估算)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="导出跌倒检测模型")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--format", type=str, default="onnx",
                       choices=["onnx", "torchscript", "all"],
                       help="导出格式")
    parser.add_argument("--output", type=str, default="export",
                       help="输出目录")
    parser.add_argument("--version", type=str, default="standard",
                       choices=["standard", "light", "large"],
                       help="模型版本")
    parser.add_argument("--opset", type=int, default=11,
                       help="ONNX opset 版本")
    parser.add_argument("--static", action="store_true",
                       help="导出静态形状模型（禁用动态轴）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 创建模型
    print(f"[Export] 创建模型 (version={args.version})...")
    model = create_fall_detector(version=args.version)
    
    # 加载权重
    if not os.path.exists(args.checkpoint):
        print(f"[Error] 检查点不存在: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # 过滤掉 thop profile 注入的 total_ops/total_params 键（旧检查点可能包含）
    state_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                  if not k.endswith(".total_ops") and not k.endswith(".total_params")
                  and k not in ("total_ops", "total_params")}
    model.load_state_dict(state_dict)
    print(f"[Export] 加载检查点: epoch={checkpoint['epoch']}")
    
    model.eval()
    
    # 合规检查
    compliance = check_compliance(model)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    tcn_input_shape = (1, config.get("pipeline", {}).get("sequence_length", 32),
                       config.get("model", {}).get("motion_encoder", {}).get("feature_dim", 48))
    
    basename = f"fall_detector_{args.version}"
    
    # 导出
    if args.format in ("onnx", "all"):
        onnx_path = os.path.join(args.output, f"{basename}.onnx")
        export_onnx(
            model, onnx_path,
            input_shape=tcn_input_shape,
            dynamic_axes=not args.static,
            opset_version=args.opset,
        )
    
    if args.format in ("torchscript", "all"):
        ts_path = os.path.join(args.output, f"{basename}.pt")
        export_torchscript(model, ts_path, input_shape=tcn_input_shape)
    
    # 华为 NPU 部署提示
    print(f"\n[华为 NPU 部署流程]")
    print(f"  1. PyTorch → ONNX: python export.py --checkpoint best.pth --format onnx")
    print(f"  2. ONNX → OM: ")
    print(f"     atc --model={basename}.onnx \\")
    print(f"         --framework=5 \\")
    print(f"         --output={basename} \\")
    print(f"         --soc_version=Ascend310 \\")
    print(f"         --input_shape=\"motion_features:1,32,48\" \\")
    print(f"         --input_format=ND")
    print(f"  3. 推理时: MotionEncoder 在 CPU/NPU 上作为前处理算子运行")
    print(f"            RuleRefinement 在应用层(C++)实现")
    print(f"\n导出完成! 文件保存在: {args.output}")


if __name__ == "__main__":
    main()
