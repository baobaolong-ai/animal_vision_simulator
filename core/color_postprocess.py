"""
core/color_postprocess.py
在色彩映射之后，对图像进行饱和度微调和亮度对比度增强。
仅作用于 HSV 空间，不改变色相，旨在恢复因二色视投影损失的视觉对比度。
"""
import cv2
import numpy as np

def apply_color_postprocess(image: np.ndarray, saturation_boost: float = 1.2) -> np.ndarray:
    """
    增强色彩映射后图像的视觉对比度。
    :param image: 色彩映射后的图像，uint8 RGB
    :param saturation_boost: 饱和度倍增系数，>1 则增强，1 则不变
    :return: 处理后的图像，uint8 RGB
    """
    # 转到 HSV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # 分离通道
    h, s, v = cv2.split(hsv)
    # 饱和度增强
    s = np.clip(s * saturation_boost, 0, 255)
    # 亮度对比度增强：对 V 通道应用 CLAHE
    v_u8 = v.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v_u8)
    # 合并
    hsv_enhanced = cv2.merge([h, s, v_eq.astype(np.float32)])
    # 转回 RGB
    result = cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result