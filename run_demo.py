#!/usr/bin/env python3
"""
动物视觉模拟器 - 命令行工具
用法示例：
    python run_demo.py --image my_photo.jpg --animal dog --output result.jpg
    python run_demo.py --image my_photo.jpg --animal cat --acuity 0.8 --luminance 0.6
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 将项目根目录加入 sys.path，确保可以导入 core 和 models
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.color_mapping import generate_3dlut, apply_3dlut
from core.pipeline import process_image_with_lut
from models.animal_vision_profile import load_all_profiles


def main():
    # 加载所有动物配置
    profiles = load_all_profiles()
    animal_choices = list(profiles.keys())

    parser = argparse.ArgumentParser(
        description="动物视觉模拟器 - 命令行工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"可选的动物 ID: {', '.join(animal_choices)}"
    )
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--animal", required=True, choices=animal_choices, help="动物 ID")
    parser.add_argument("--output", default="output.jpg", help="输出图片路径 (默认 output.jpg)")
    parser.add_argument("--acuity", type=float, default=0.7, help="视锐度补偿系数 (0-1)，默认 0.7")
    parser.add_argument("--luminance", type=float, default=0.5, help="明暗补偿系数 (0-1)，默认 0.5")
    parser.add_argument("--saturation", type=float, default=1.2, help="饱和度增强系数，默认 1.2")
    parser.add_argument("--no-lut-cache", action="store_true",
                        help="不使用预生成的 LUT 文件，动态生成 (速度较慢)")

    args = parser.parse_args()

    # 检查输入图像
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误：找不到图片文件 {args.image}")
        return

    # 读取图像
    print(f"读取图像: {args.image}")
    image = np.array(Image.open(image_path).convert("RGB"))

    # 获取动物配置
    profile = profiles[args.animal]

    # 1. 色彩映射
    if args.animal == "human":
        processed = image.copy()
    else:
        if args.no_lut_cache:
            print(f"动态生成 {profile.name} 的 LUT ...")
            lut = generate_3dlut(args.animal)
        else:
            lut_dir = Path(__file__).resolve().parent / "config" / "luts"
            lut_path = lut_dir / f"{args.animal}.npy"
            if lut_path.exists():
                print(f"加载预生成 LUT: {lut_path}")
                lut = np.load(lut_path)
            else:
                print(f"未找到 LUT 文件，动态生成 {profile.name} 的 LUT ...")
                lut = generate_3dlut(args.animal)
        processed = apply_3dlut(image, lut)

    # 2. 完整处理管线 (明暗适应、视锐度、特殊感知)
    print("执行视觉模拟管线 ...")
    processed = process_image_with_lut(
        processed,
        profile,
        acuity_adaptation=args.acuity,
        luminance_adaptation=args.luminance,
        saturation_boost=args.saturation
    )

    # 保存结果
    output_path = Path(args.output)
    Image.fromarray(processed).save(output_path)
    print(f"处理完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    main()