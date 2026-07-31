# models/app_state.py
from PySide6.QtCore import QObject, Signal
from models.animal_vision_profile import AnimalVisionProfile


class AppState(QObject):
    """应用全局状态管理"""
    animal_changed = Signal(AnimalVisionProfile)  # 动物切换信号

    def __init__(self, profiles: dict[str, AnimalVisionProfile]):
        super().__init__()
        self._profiles = profiles
        self._current_animal_id = "human"  # 默认人类

    @property
    def current_profile(self) -> AnimalVisionProfile:
        return self._profiles[self._current_animal_id]

    def set_animal(self, animal_id: str):
        """切换当前动物，发射通知信号"""
        if animal_id in self._profiles and animal_id != self._current_animal_id:
            self._current_animal_id = animal_id
            self.animal_changed.emit(self.current_profile)