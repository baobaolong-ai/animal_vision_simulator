# Animal Vision Simulator · Core Algorithm Library

透过动物的眼睛看世界——基于视觉生理学的图像处理管线。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

## 项目简介

本仓库是**动物视觉模拟器**的核心算法库，提供了完整的动物视觉模拟图像处理管线。它能够将普通可见光照片，根据10种动物的视觉生理参数，转换为它们眼中的世界。

**核心能力**：

- **色彩映射**：基于 Machado 2009 矩阵的二色视、全色盲及紫外偏移色觉模拟。
- **暗适应增益**：CLAHE 局部对比度增强，模拟暗视觉灵敏度和反光膜光晕。
- **高光抑制**：提亮暗部的同时，完整保留原始高光区域的纹理和层次。
- **视锐度模拟**：空间变化高斯模糊，实现中央清晰、外周模糊的视网膜采样效果。
- **神经适应**：模拟大脑补偿的可调参数（视锐度补偿、明暗补偿、饱和度增强）。

> **请注意**：本仓库仅包含**算法层**，不含任何图形界面 (GUI) 代码。如需完整的桌面应用（含扫描线对比、动态仪表盘、动物头像等交互功能），请访问[ Animal Vision Simulator Releases ](https://github.com/baobaolong-ai/animal_vision_simulator/releases)下载免费安装包。

## 特点

- **配置驱动**：所有动物视觉参数通过 JSON 配置文件管理，新增动物无需修改代码。
- **模块化设计**：各算法模块独立，通过统一接口 (`image, profile → image`) 自由组合。
- **多入口支持**：提供命令行工具、Gradio 两种方式运行。
- **科学严谨**：算法均基于已发表的视觉生理学研究，参数有明确的生理学依据。

## 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/yourname/animal-vision-core.git
cd animal-vision-core
pip install -r requirements.txt
```

### 2. 生成 LUT 文件（首次运行必需）

```bash
python tools/generate_luts.py
```

这一步会为所有动物生成三维查找表 (64×64×64)，保存在 `config/luts/` 目录下，总大小约 6 x 10 MB。此过程只需执行一次。

### 3. 选择一种方式运行

#### 命令行

```bash
python run_demo.py --image examples/sample.jpg --animal dog --output result.jpg
```

**参数说明**：

| 参数             | 说明             | 默认值         |
| ---------------- | ---------------- | -------------- |
| `--image`      | 输入图片路径     | (必需)         |
| `--animal`     | 动物 ID          | (必需)         |
| `--output`     | 输出图片路径     | `output.jpg` |
| `--acuity`     | 视锐度补偿 (0-1) | 0.7            |
| `--luminance`  | 明暗补偿 (0-1)   | 0.5            |
| `--saturation` | 饱和度增强系数   | 1.2            |

#### Gradio 本地 Web 界面

```bash
pip install gradio
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，可上传图片、选择动物、调节参数并实时预览。

## 可用动物列表

| 动物 ID             | 名称   | 色觉类型          |
| ------------------- | ------ | ----------------- |
| `human`           | 人     | 三色视 (基准)     |
| `dog`             | 狗     | 二色视 (蓝-黄)    |
| `cat`             | 猫     | 二色视 (蓝-绿)    |
| `horse`           | 马     | 二色视 (蓝-绿)    |
| `rabbit`          | 兔子   | 二色视 (蓝-绿)    |
| `eagle`           | 鹰     | 四色视 (含紫外)   |
| `owl`             | 猫头鹰 | 近全色盲          |
| `goldfish`        | 金鱼   | 四色视 (含紫外)   |
| `squirrel_monkey` | 松鼠猴 | 二色视 (M 锥缺失) |
| `red_kangaroo`    | 红袋鼠 | S 锥功能退化      |

## 项目结构

```
animal-vision-core/
├── README.md                       # 本文件
├── LICENSE                         # MIT 许可证
├── requirements.txt                # 核心依赖
├── run_demo.py                     # 命令行演示工具
├── app.py                          # Gradio Web 界面
├── streamlit_app.py                # Streamlit 在线演示
│
├── core/                           # 核心算法模块
│   ├── color_mapping.py            # 色彩映射 (Machado + 3D LUT)
│   ├── luminance_adaptation.py     # 暗适应增益 (CLAHE + 高光保护)
│   ├── acuity_simulation.py        # 视锐度模拟 (空间变化模糊)
│   ├── pipeline.py                 # 处理管线 (串联所有模块)
│   └── color_postprocess.py        # 色彩后处理 (饱和度/对比度微调)
│
├── models/                         # 数据模型
│   └── animal_vision_profile.py    # 动物视觉配置的数据类及加载函数
│
├── config/                         # 配置文件
│   ├── animals/                    # 10种动物的 JSON 配置
│   │   ├── human.json
│   │   ├── dog.json
│   │   └── ...
│   └── luts/                       # 预生成的三维查找表 (运行 generate_luts.py 后生成)
│       ├── human.npz
│       ├── dog.npz
│       └── ...
│
├── tools/                          # 辅助脚本
│   └── generate_luts.py            # 生成 LUT 文件
│
└── images/                      	# 示例图片
```

## API 快速参考

核心处理管线只需一个函数调用：

```python
import numpy as np
from PIL import Image
from core.pipeline import process_image_with_lut
from core.color_mapping import generate_3dlut, apply_3dlut
from models.animal_vision_profile import load_all_profiles

# 加载配置和 LUT
profiles = load_all_profiles()
lut = generate_3dlut("dog")

# 读取图片并处理
image = np.array(Image.open("photo.jpg").convert("RGB"))
processed = apply_3dlut(image, lut)  # 色彩映射
processed = process_image_with_lut(
    processed, profiles["dog"],
    acuity_adaptation=0.7,
    luminance_adaptation=0.5,
    saturation_boost=1.2
)  # 完整管线 (明暗、锐度、特殊感知)

# 保存结果
Image.fromarray(processed).save("dog_view.jpg")
```

## 科学依据

本项目的算法主要基于以下研究工作：

- **Machado, G. M., et al. (2009).** *A Physiologically-based Model for Simulation of Color Vision Deficiency.* IEEE TVCG.
- **Brettel, H., Viénot, F., & Mollon, J. D. (1997).** *Computerized simulation of color appearance for dichromats.* JOSA A.
- **Viénot, F., Brettel, H., & Mollon, J. D. (1999).** *Digital video colourmaps for checking the legibility of displays by dichromats.* Color Research & Application.
- **Stockman, A., & Sharpe, L. T. (2000).** *Spectral sensitivities of the middle- and long-wavelength sensitive cones.* Vision Research.

动物视觉参数（视锐度、视野、暗光灵敏度等）参考了：

- **Land, M. F., & Nilsson, D.-E. (2012).** *Animal Eyes.* Oxford University Press.
- **Jacobs, G. H. (1993).** *The distribution and nature of colour vision among the mammals.* Biological Reviews.
- **Jacobs, G. H. (2009).** *Evolution of colour vision in mammals.* Phil. Trans. R. Soc. B.
- **Peichl, L. (2005).** *Diversity of mammalian photoreceptor properties.* The Anatomical Record.

## 许可证

本项目核心算法库采用 **MIT 许可证**。您可以自由使用、修改和分发这些代码，但需保留原始版权声明。

## 联系与反馈

如果您在使用过程中遇到问题，或有任何建议，欢迎通过 GitHub Issues 交流。

---

*透过动物的眼睛，看见更广阔的世界。*
