# models/animal_vision_profile.py
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ColorVision:
    type: str  # "trichromatic" | "dichromatic" | "tetrachromatic"
    sensitive_spectrum: List[str]
    description: str


@dataclass
class VisualAcuity:
    equivalent_snellen: str
    description: str


@dataclass
class FieldOfView:
    total_degrees: float
    binocular_overlap_degrees: float
    blind_spot_positions: List[str]
    description: str


@dataclass
class TemporalPerception:
    flicker_fusion_hz: float
    description: str


@dataclass
class LuminanceSensitivity:
    scotopic_enhancement_factor: float
    has_tapetum_lucidum: bool
    description: str


@dataclass
class AnimalVisionProfile:
    id: str
    name: str
    scientific_name: str
    color_vision: ColorVision
    visual_acuity: VisualAcuity
    field_of_view: FieldOfView
    temporal_perception: TemporalPerception
    luminance_sensitivity: LuminanceSensitivity
    special_perceptions: List[str]
    ecological_summary: str
    acuity_params: dict = field(default_factory=dict)
    layman_description: str = ""

    @classmethod
    def from_json(cls, filepath: Path) -> "AnimalVisionProfile":
        """从JSON文件加载动物视觉配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            id=data["id"],
            name=data["name"],
            scientific_name=data["scientific_name"],
            color_vision=ColorVision(**data["color_vision"]),
            visual_acuity=VisualAcuity(**data["visual_acuity"]),
            field_of_view=FieldOfView(**data["field_of_view"]),
            temporal_perception=TemporalPerception(**data["temporal_perception"]),
            luminance_sensitivity=LuminanceSensitivity(**data["luminance_sensitivity"]),
            special_perceptions=data["special_perceptions"],
            ecological_summary=data["ecological_summary"],
            acuity_params=data.get("acuity_params", {}),
            layman_description=data.get("layman_description", "")
        )


# 动物配置注册表
ANIMAL_CONFIG_DIR = Path(__file__).parent.parent / "config" / "animals"


def load_all_profiles() -> dict[str, AnimalVisionProfile]:
    """加载所有动物视觉配置"""
    profiles = {}
    for json_file in ANIMAL_CONFIG_DIR.glob("*.json"):
        profile = AnimalVisionProfile.from_json(json_file)
        profiles[profile.id] = profile
    return profiles