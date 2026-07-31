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

    foveal_deg = params.get('foveal_radius_deg', 2)
    max_sigma_ratio = params.get('max_blur_sigma_ratio', 0.0)
    exponent = params.get('falloff_exponent', 2.0)

    if max_sigma_ratio <= 0.0 or foveal_deg >= 60:
        return image

    H, W = image.shape[:2]
    central_ratio = foveal_deg / 60.0
    max_sigma = W * max_sigma_ratio

    # 归一化距离图
    y, x = np.ogrid[:H, :W]
    cx, cy = W / 2, H / 2
    dist_x = (x - cx) / (W / 2)
    dist_y = (y - cy) / (H / 2)
    dist_norm = np.sqrt(dist_x**2 + dist_y**2) / 1.414
    dist_norm = np.clip(dist_norm, 0, 1)

    over_dist = np.clip((dist_norm - central_ratio) / (1.0 - central_ratio), 0, 1)
    sigma_map = max_sigma * (over_dist ** exponent)

    if max_sigma < 1.0:
        return image

    num_layers = 4
    sigma_layers = np.linspace(0, max_sigma, num_layers)

    # 生成多层模糊图像
    blurred_images = []
    for s in sigma_layers:
        if s <= 0.5:
            blurred_images.append(image.astype(np.float32))
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=s, sigmaY=s)
            blurred_images.append(blurred.astype(np.float32))

    layers = np.stack(blurred_images, axis=0)  # float32

    layer_idx = (sigma_map / max_sigma) * (num_layers - 1)
    layer_idx = np.clip(layer_idx, 0, num_layers - 1)
    idx_low = np.floor(layer_idx).astype(np.int32)
    idx_high = np.minimum(idx_low + 1, num_layers - 1)
    alpha = (layer_idx - idx_low)[..., np.newaxis]

    H_grid, W_grid = np.ogrid[:H, :W]
    low_vals = layers[idx_low, H_grid, W_grid]
    high_vals = layers[idx_high, H_grid, W_grid]

    interpolated = (1 - alpha) * low_vals + alpha * high_vals

    # 中央锐利保护：sigma < 0.5 的区域直接使用原图
    sharp_mask = (sigma_map < 0.5)[..., np.newaxis]
    result = np.where(sharp_mask, image.astype(np.float32), interpolated)

    return np.clip(result, 0, 255).astype(np.uint8)