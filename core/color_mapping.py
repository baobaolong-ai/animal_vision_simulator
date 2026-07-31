"""
core/color_mapping.py
使用 colour-science 库的 Machado 2009 矩阵生成 3D LUT，
并对特殊动物进行自定义色彩映射。
"""
import numpy as np
import colour

# 基础工具
def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (1.0 / 2.4) - 0.055)

RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# 动物映射
ANIMAL_VISION = {
    'human':            ('human', None),
    'dog':              ('machado', 'Protanomaly'),
    'cat':              ('machado', 'Protanomaly'),
    'horse':            ('machado', 'Protanomaly'),
    'rabbit':           ('machado', 'Protanomaly'),   # 多数研究支持缺失L锥
    'squirrel_monkey':  ('machado', 'Deuteranomaly'),
    'red_kangaroo':     ('machado', 'Tritanomaly'),
    'eagle':            ('eagle', None),
    'owl':              ('grayscale', None),
    'goldfish':         ('goldfish', None),
}

def generate_3dlut(animal_id: str, size: int = 64) -> np.ndarray:
    if animal_id not in ANIMAL_VISION:
        raise ValueError(f"未知的动物ID: '{animal_id}'")

    r = np.linspace(0, 1, size, dtype=np.float32)
    g = np.linspace(0, 1, size, dtype=np.float32)
    b = np.linspace(0, 1, size, dtype=np.float32)
    R, G, B = np.meshgrid(r, g, b, indexing='ij')
    colors_linear = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)

    method, param = ANIMAL_VISION[animal_id]

    if method == 'human':
        transformed_linear = colors_linear
    elif method == 'machado':
        M = colour.blindness.matrix_cvd_Machado2009(param, 1.0)
        transformed_linear = colors_linear @ M.T
        transformed_linear = np.clip(transformed_linear, 0.0, 1.0)
    elif method == 'grayscale':
        xyz = colors_linear @ RGB_TO_XYZ.T
        Y = xyz[:, 1:2]
        transformed_linear = np.repeat(Y, 3, axis=1)
    elif method == 'goldfish':
        transformed_linear = colors_linear.copy()
        transformed_linear[:, 0] *= 0.3
        transformed_linear[:, 1] *= 1.3
        transformed_linear[:, 2] *= 1.8
        y_orig = np.dot(colors_linear, RGB_TO_XYZ[:, 1])
        y_new = np.dot(transformed_linear, RGB_TO_XYZ[:, 1])
        mask = y_new > 1e-6
        transformed_linear[mask] *= (y_orig[mask] / y_new[mask]).reshape(-1, 1)
        transformed_linear = np.clip(transformed_linear, 0.0, 1.0)
    elif method == 'eagle':
        transformed_linear = colors_linear  # 未来可叠加紫外示意
    else:
        transformed_linear = colors_linear

    lut = linear_to_srgb(transformed_linear)
    return lut.reshape(size, size, size, 3)

def apply_3dlut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """三线性插值应用 LUT"""
    size = lut.shape[0]
    img_float = image.astype(np.float32) / 255.0 * (size - 1)
    x0 = np.floor(img_float).astype(np.int32)
    x1 = np.minimum(x0 + 1, size - 1)
    dx = img_float - x0.astype(np.float32)

    r0, g0, b0 = x0[..., 0], x0[..., 1], x0[..., 2]
    r1, g1, b1 = x1[..., 0], x1[..., 1], x1[..., 2]

    c000 = lut[r0, g0, b0]
    c100 = lut[r1, g0, b0]
    c010 = lut[r0, g1, b0]
    c110 = lut[r1, g1, b0]
    c001 = lut[r0, g0, b1]
    c101 = lut[r1, g0, b1]
    c011 = lut[r0, g1, b1]
    c111 = lut[r1, g1, b1]

    dr = dx[..., 0:1]
    dg = dx[..., 1:2]
    db = dx[..., 2:3]

    c00 = c000 * (1 - dr) + c100 * dr
    c01 = c001 * (1 - dr) + c101 * dr
    c10 = c010 * (1 - dr) + c110 * dr
    c11 = c011 * (1 - dr) + c111 * dr
    c0 = c00 * (1 - dg) + c10 * dg
    c1 = c01 * (1 - dg) + c11 * dg
    result = c0 * (1 - db) + c1 * db

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)