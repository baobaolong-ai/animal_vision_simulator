#!/usr/bin/env python3
"""
动物视觉模拟器 - Gradio Web 界面
用法：
    pip install gradio
    python app.py
然后在浏览器中打开 http://127.0.0.1:7860
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.color_mapping import generate_3dlut, apply_3dlut
from core.pipeline import process_image_with_lut
from models.animal_vision_profile import load_all_profiles

# ── 预加载资源 ──────────────────────────────────────────────
profiles = load_all_profiles()
animal_choices = [(profile.name, animal_id) for animal_id, profile in profiles.items()]
animal_ids = [aid for _, aid in animal_choices]

# 生成/加载 LUT
luts = {}
lut_dir = Path(__file__).resolve().parent / "config" / "luts"
for animal_id in animal_ids:
    if animal_id == "human":
        continue
    lut_path = lut_dir / f"{animal_id}.npy"
    if lut_path.exists():
        luts[animal_id] = np.load(lut_path)
    else:
        luts[animal_id] = generate_3dlut(animal_id)

# ── 核心处理函数 ────────────────────────────────────────────
def simulate(image, animal_id, acuity, luminance, saturation):
    if image is None:
        return None

    # 限制最大尺寸，避免内存溢出
    h, w = image.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = Image.fromarray(image).resize((new_w, new_h))
        image = np.array(image)

    profile = profiles[animal_id]

    # 色彩映射
    if animal_id == "human":
        processed = image.copy()
    else:
        processed = apply_3dlut(image, luts[animal_id])

    # 完整管线
    processed = process_image_with_lut(
        processed,
        profile,
        acuity_adaptation=acuity,
        luminance_adaptation=luminance,
        saturation_boost=saturation
    )
    return Image.fromarray(processed)

# ── 构建 Gradio 界面 ─────────────────────────────────────────
with gr.Blocks(title="动物视觉模拟器") as demo:
    gr.Markdown("""
    # 🐾 动物视觉模拟器
    
    上传一张照片，选择一种动物，立刻看到它们眼中的世界。
    核心算法基于 **Machado 2009** 色觉模型及视觉生理学研究。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="📷 上传照片")
            animal_dropdown = gr.Dropdown(
                choices=[(name, aid) for name, aid in animal_choices],
                value="dog",
                label="🎯 选择动物"
            )
            with gr.Accordion("🧠 神经适应调节", open=True):
                acuity_slider = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="视锐度补偿")
                luminance_slider = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="明暗补偿")
                saturation_slider = gr.Slider(0.5, 2.0, 1.2, step=0.1, label="饱和度增强")

            submit_btn = gr.Button("🔍 生成动物视角", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="✨ 模拟结果")

    # 绑定事件
    submit_btn.click(
        fn=simulate,
        inputs=[image_input, animal_dropdown, acuity_slider, luminance_slider, saturation_slider],
        outputs=image_output
    )

    # 下载按钮复用同一个模拟函数并触发下载
    gr.Markdown("💡 提示：右键点击右侧生成的图片即可保存。")

# ── 启动 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()