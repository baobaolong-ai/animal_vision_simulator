"""
core/luminance_adaptation.py
明暗适应模拟：使用限制对比度自适应直方图均衡（CLAHE）在亮度通道上提亮暗部，
并通过视觉侧抑制模拟（基于距离变换的渐进掩膜）保护高光区域纹理。
反光膜光晕仅微弱叠加。
"""
import cv2
import numpy as np
import math


def _create_distance_based_glow_mask(binary_mask, max_decay_distance=30):
    """
    基于距离变换生成外部渐变掩膜，用于模拟视觉侧抑制的光晕扩散效果。
    
    该掩膜具有以下特性：
    - 高光区域内部（原掩膜为1的区域）保持为 1，纹理完全保留。
    - 外部过渡带：在 max_decay_distance 像素范围内从 1 线性衰减到 0。
    - 衰减带以外：值为 0，不进行任何高光保护。
    
    :param binary_mask: 二值高光掩膜，形状 (H, W)，值域 {0, 1}
    :param max_decay_distance: 光晕向外扩散的最大像素距离
    :return: 渐进掩膜，形状 (H, W)，值域 [0, 1]
    """
    # 对反向掩膜做欧几里得距离变换：计算非高光区域每个像素到最近高光边界的距离
    mask_inv = (1.0 - binary_mask).astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)

    # 将距离映射为 [0, 1]：距离为 0 处值为 1，距离越远值越小
    glow = np.clip(1.0 - dist / max_decay_distance, 0.0, 1.0)

    # 确保原始高光区域内部值为 1（避免数值精度导致偏差）
    glow[binary_mask > 0.5] = 1.0

    return glow


def apply_luminance_adaptation(image: np.ndarray, profile) -> np.ndarray:
    """
    模拟暗视觉增强及反光膜光晕。
    
    :param image: 输入图像，uint8 RGB（色彩映射之后）
    :param profile: AnimalVisionProfile 对象
    :return: 处理后的图像，uint8 RGB
    """
    params = profile.luminance_sensitivity
    factor = params.scotopic_enhancement_factor
    has_tapetum = params.has_tapetum_lucidum

    # 若无暗视觉增强且无反光膜，直接返回原图
    if factor <= 1.0 and not has_tapetum:
        return image

    # ---------- 暗部增强 + 视觉侧抑制模拟 ----------
    if factor > 1.0:
        # 转换到 LAB 空间，仅处理 L 通道（避免影响色相与饱和度）
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 保存原始 L 通道（色彩映射后的亮度），作为高光保护的基准
        l_orig = l_channel.astype(np.float32)

        # --- 1. 基于原始亮度生成二值高光掩膜 ---
        # 只有原始亮度超过阈值的区域才被视为需要保护的高光
        highlight_threshold = 200
        binary_mask = (l_orig > highlight_threshold).astype(np.float32)

        # --- 2. 距离变换生成外部渐变掩膜 ---
        # 该掩膜在高光内部为 1，外部在指定像素范围内线性衰减到 0
        max_decay_distance = 30
        M = _create_distance_based_glow_mask(binary_mask, max_decay_distance)

        # --- 3. CLAHE 增强（全图执行，不跳过任何区域） ---
        clip_limit = 2.0 + 3.0 * math.log2(factor)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # --- 4. 带强度系数的掩膜混合 ---
        # k = 0.8 表示高光区域保留 80% 的原始纹理，20% 来自 CLAHE 增强，保持适度亮感
        k = 0.8
        l_new = (l_clahe.astype(np.float32) * (1.0 - M * k) +
                 l_orig * (M * k))
        l_new = l_new.clip(0, 255).astype(np.uint8)

        # 合并 LAB 并转回 RGB
        lab_enhanced = cv2.merge([l_new, a_channel, b_channel])
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # ---------- 反光膜光晕（物理散射模拟） ----------
    if has_tapetum:
        # 提取高亮区域 (亮度 > 0.7)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        highlight = (gray > 0.7).astype(np.float32)

        # 大尺度高斯模糊模拟光散射，核大小取图像高度的 5%
        bloom_size = max(21, int(image.shape[0] * 0.05) * 2 + 1)
        bloomed = cv2.GaussianBlur(image.astype(np.float32) * highlight[..., np.newaxis],
                                   (bloom_size, bloom_size), 0)

        # 滤色混合：result = 1 - (1 - base) × (1 - bloom × 0.2)
        img_float = image.astype(np.float32) / 255.0
        bloomed /= 255.0
        result = 1.0 - (1.0 - img_float) * (1.0 - bloomed * 0.2)
        image = np.clip(result * 255, 0, 255).astype(np.uint8)

    return image