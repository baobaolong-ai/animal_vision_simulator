"""
core/pipeline.py
动物视觉模拟管线：色彩映射 → 色彩后处理 → 明暗适应 → 视锐度模拟。
支持神经适应系数。
"""
import copy
import numpy as np
from models.animal_vision_profile import AnimalVisionProfile
from core.luminance_adaptation import apply_luminance_adaptation
from core.acuity_simulation import apply_acuity_simulation
from core.color_postprocess import apply_color_postprocess

def process_image_with_lut(image: np.ndarray,
                           profile: AnimalVisionProfile,
                           acuity_adaptation: float = 0.7,
                           luminance_adaptation: float = 0.5,
                           saturation_boost: float = 1.2) -> np.ndarray:
    """
    完整的视觉模拟管线。
    :param image: 已应用色彩映射的图像（uint8 RGB）
    :param profile: 动物视觉配置
    :param acuity_adaptation: 视锐度神经适应系数
    :param luminance_adaptation: 明暗适应系数
    :param saturation_boost: 饱和度增强系数
    :return: 最终处理图像
    """

    processed = image.copy()
    # 1. 色彩后处理（恢复部分视觉对比度）
    processed = apply_color_postprocess(processed, saturation_boost)

    # 2. 明暗适应（CLAHE 强度由 luminance_adaptation 控制）
    base_factor = profile.luminance_sensitivity.scotopic_enhancement_factor
    has_tapetum = profile.luminance_sensitivity.has_tapetum_lucidum

    if base_factor > 1.0 or has_tapetum:
        # luminance_adaptation: 用户滑块值，范围 [0, 1]
        # 调整因子：0 时降至 1.0（无增强），1 时保持原 base_factor
        adjusted_factor = 1.0 + (base_factor - 1.0) * luminance_adaptation
        
        # 创建副本避免污染原始配置
        profile_copy = copy.copy(profile)
        profile_copy.luminance_sensitivity = copy.copy(profile.luminance_sensitivity)
        profile_copy.luminance_sensitivity.scotopic_enhancement_factor = adjusted_factor
        processed = apply_luminance_adaptation(processed, profile_copy)

    # 3. 视锐度模拟
    if profile.acuity_params.get('max_blur_sigma_ratio', 0) > 0:
        original_max_sigma = profile.acuity_params['max_blur_sigma_ratio']
        adjusted_sigma = original_max_sigma * (1.0 - acuity_adaptation)
        if adjusted_sigma > 0:
            profile_copy = copy.copy(profile)
            profile_copy.acuity_params = dict(profile.acuity_params)
            profile_copy.acuity_params['max_blur_sigma_ratio'] = adjusted_sigma
            processed = apply_acuity_simulation(processed, profile_copy)

    return processed