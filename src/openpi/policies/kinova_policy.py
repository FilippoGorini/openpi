"""
Kinova Gen3 6-DoF + Robotiq 2F-85 policy transforms for pi0_base.

Modeled on examples/ur5/README.md — UR5e is also 6-DoF + 1 gripper = 7 action dims,
and pi0_base ships UR5e normalization stats (asset_id="ur5e") that we reuse here.

Observation expected from the ROS 2 client:
  joints       float32 (6,)     arm joint positions in radians (joint_1..joint_6)
  gripper      float32 (1,)     knuckle joint position in radians (0=open, ~0.8=closed)
  base_rgb     uint8   (H,W,3)  external/overhead camera — any resolution, resized server-side
  wrist_rgb    uint8   (H,W,3)  wrist camera — any resolution, resized server-side
  prompt       str              language instruction

Action returned to the ROS 2 client:
  actions      float32 (horizon, 7)  absolute positions: cols 0:6 = arm joints, col 6 = gripper
                                     (AbsoluteActions transform is applied server-side)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Inference: ROS2 client sends joints (6,) + gripper (1,) separately.
        # Training: repack transform delivers observation.state as a single "state" (7,).
        if "joints" in data:
            state = np.concatenate([data["joints"], data["gripper"]])  # (7,)
        else:
            state = np.asarray(data["state"])  # (7,) already concatenated

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No right wrist on Kinova — zero-padded and masked out
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
