import os
from .clip_encoder import AutoVisionTower, CLIPVisionTower, CLIPVisionTowerS2, SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower_name = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    # use_s2 = getattr(vision_tower_cfg, 's2', False)
    if "siglip" in vision_tower_name.lower():
        # return AutoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return SiglipVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)
    elif vision_tower_name.startswith("openai") or vision_tower_name.startswith("laion") or "ShareGPT4V" in vision_tower_name:
        return CLIPVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)
    else:
        return AutoVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower_name}')

if __name__ == "__main__":
    print("Building Vision Encoder.")
