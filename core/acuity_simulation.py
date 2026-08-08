"""
core/acuity_simulation.py
视锐度模拟：中央清晰、外周模糊，中央硬保护。
"""
import cv2
import numpy as np

def apply_acuity_simulation(image: np.ndarray, profile) -> np.ndarray:
    params = profile.acuity_params
    if not params:
        return image

    foveal_deg = params.get('foveal_radius_deg', 2)       # 中央清晰区视角半径（度）
    max_sigma_ratio = params.get('max_blur_sigma_ratio', 0.0)
    exponent = params.get('falloff_exponent', 2.0)

    # 无效参数直接返回原图
    if max_sigma_ratio <= 0.0 or foveal_deg >= 60:
        return image

    H, W = image.shape[:2]

    # ---- 数学原理严格映射 ----
    # 归一化距离阈值：τ = r / (30°·√2)，保证σ=0区域严格对应视角半径r
    central_ratio = foveal_deg / (30.0 * np.sqrt(2))      # ← 修正点1：原为 foveal_deg/60.0
    max_sigma = W * max_sigma_ratio                        # 最大模糊标准差（像素）

    # 归一化距离图（d ∈ [0,1]）
    y, x = np.ogrid[:H, :W]
    cx, cy = W / 2, H / 2
    dist_x = (x - cx) / (W / 2)
    dist_y = (y - cy) / (H / 2)
    # d = sqrt(dx²+dy²)/√2，四角为1，水平/垂直边缘为~0.707
    dist_norm = np.sqrt(dist_x**2 + dist_y**2) / np.sqrt(2)  # 用 √2 替代 1.414 保持精度
    dist_norm = np.clip(dist_norm, 0, 1)

    # 过渡变量：d_over = max(0, (d-τ)/(1-τ))
    over_dist = np.clip((dist_norm - central_ratio) / (1.0 - central_ratio), 0, 1)
    sigma_map = max_sigma * (over_dist ** exponent)        # 幂次衰减

    # 最大模糊小于1像素时不做处理（速度优化）
    if max_sigma < 1.0:
        return image

    # ---- 分层近似（工程加速） ----
    num_layers = 4
    sigma_layers = np.linspace(0, max_sigma, num_layers)

    # 生成多层模糊图像
    blurred_images = []
    for s in sigma_layers:
        if s == 0:   # ← 修正点2：仅σ=0时保留原图，其余严格应用高斯模糊
            blurred_images.append(image.astype(np.float32))
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=s, sigmaY=s)
            blurred_images.append(blurred.astype(np.float32))
    layers = np.stack(blurred_images, axis=0)  # (4, H, W)

    # 逐像素插值系数计算
    layer_idx = (sigma_map / max_sigma) * (num_layers - 1)
    layer_idx = np.clip(layer_idx, 0, num_layers - 1)
    idx_low = np.floor(layer_idx).astype(np.int32)
    idx_high = np.minimum(idx_low + 1, num_layers - 1)
    alpha = (layer_idx - idx_low)[..., np.newaxis]        # 插值权重

    # 高级索引取对应图层像素（向量化）
    H_grid, W_grid = np.ogrid[:H, :W]
    low_vals = layers[idx_low, H_grid, W_grid]
    high_vals = layers[idx_high, H_grid, W_grid]

    interpolated = (1 - alpha) * low_vals + alpha * high_vals

    # ---- 中央硬保护（精确边界） ----
    # 保护条件：严格满足 d ≤ τ 的所有像素（sigma=0），避免模糊污染
    sharp_mask = (dist_norm <= central_ratio)  # ← 修正点3：原为 sigma_map < 0.5
    sharp_mask = sharp_mask[..., np.newaxis]   # 扩展通道维度
    result = np.where(sharp_mask, image.astype(np.float32), interpolated)

    return np.clip(result, 0, 255).astype(np.uint8)