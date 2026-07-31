import numpy as np
from PySide6.QtGui import QImage, QPixmap

def qpixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
    """将 QPixmap 转换为 RGB888 的 NumPy 数组（拷贝）"""
    qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    w, h = qimage.width(), qimage.height()
    byte_per_line = qimage.bytesPerLine()
    # 获取原始字节数据
    ptr = qimage.constBits()
    # 在 PySide6 中，constBits() 可能返回 bytes 或 memoryview
    if isinstance(ptr, memoryview):
        ptr = bytes(ptr)
    arr = np.frombuffer(ptr, dtype=np.uint8)
    # 处理可能的行填充
    if byte_per_line == w * 3:
        return arr.reshape((h, w, 3)).copy()
    else:
        arr = arr.reshape((h, byte_per_line))
        return arr[:, :w * 3].reshape((h, w, 3)).copy()

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """将 RGB888 的 NumPy 数组转换为 QPixmap"""
    h, w = arr.shape[:2]
    # 确保数组连续且类型正确
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    qimage = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)