#!/usr/bin/env python3
"""
离线生成所有动物的 3D LUT 文件。
运行方式：在项目根目录下执行 python tools/generate_luts.py
"""
import sys
sys.path.insert(0, '.')  # 确保可以导入 core 模块

import numpy as np
from pathlib import Path
from core.color_mapping import generate_3dlut
from models.animal_vision_profile import load_all_profiles

def main():
    profiles = load_all_profiles()
    lut_dir = Path("config/luts")
    lut_dir.mkdir(parents=True, exist_ok=True)

    for animal_id, profile in profiles.items():
        print(f"正在生成 {profile.name} ({animal_id}) 的 LUT...")
        try:
            lut = generate_3dlut(animal_id)
            np.save(lut_dir / f"{animal_id}.npy", lut)
            print("  完成")
        except Exception as e:
            print(f"  生成失败: {e}")

    print("\n所有 LUT 生成完毕！")

if __name__ == "__main__":
    main()