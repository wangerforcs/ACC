import numpy as np

from libero.libero.envs.bddl_base_domain import register_problem
from libero.libero.envs.problems.libero_floor_manipulation import (
    Libero_Floor_Manipulation,
)


@register_problem
class Libero_Dynamic_Floor_Manipulation(Libero_Floor_Manipulation):
    """
    Minimal example of a dynamic LIBERO environment.

    This environment reuses the standard floor-manipulation logic but injects a
    short, bounded position-dragging phase for one movable object after a
    settle period. By default the object follows a round trip: it first moves
    away from its initial position, then gradually returns.
    """

    def __init__(
        self,
        bddl_file_name,
        *args,
        moving_object_name=None,
        motion_axis="x",
        motion_initial_speed=None,
        motion_settle_steps=10,
        motion_drive_steps=30,
        motion_drive_speed=None,
        motion_end_speed=0.0,
        motion_return_enabled=False,
        motion_max_angular_speed=0.0,
        motion_amplitude=0.05,
        motion_frequency_hz=0.15,
        motion_enabled=True,
        **kwargs,
    ):
        self.moving_object_name = moving_object_name
        self.motion_axis = motion_axis
        # Keep backward compatibility with older configs that used amplitude /
        # frequency driven sinusoidal motion. If explicit initial speed is not
        # provided, derive a small velocity scale from the old parameters.
        if motion_initial_speed is None:
            motion_initial_speed = 2.0 * np.pi * float(motion_frequency_hz) * float(motion_amplitude)
        self.motion_initial_speed = float(motion_initial_speed)
        self.motion_settle_steps = max(0, int(motion_settle_steps))
        self.motion_drive_steps = max(0, int(motion_drive_steps))
        if motion_drive_speed is None:
            motion_drive_speed = self.motion_initial_speed
        self.motion_drive_speed = float(motion_drive_speed)
        self.motion_end_speed = float(motion_end_speed)
        self.motion_return_enabled = bool(motion_return_enabled)
        self.motion_max_angular_speed = float(motion_max_angular_speed)
        self.motion_amplitude = float(motion_amplitude)
        self.motion_frequency_hz = float(motion_frequency_hz)
        self.motion_enabled = bool(motion_enabled)

        self._motion_joint_name = None
        self._motion_qvel_slice = None
        self._motion_base_qpos = None
        self._post_reset_step_idx = 0
        self._velocity_injected = False
        self._drive_step_idx = 0

        super().__init__(bddl_file_name, *args, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        self._cache_motion_reference()
        self._post_reset_step_idx = 0
        self._velocity_injected = False
        self._drive_step_idx = 0

    def _cache_motion_reference(self):
        if not self.motion_enabled:
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_base_qpos = None
            return

        if self.moving_object_name is None:
            if len(self.objects_dict) == 0:
                self._motion_joint_name = None
                self._motion_qvel_slice = None
                self._motion_base_qpos = None
                return
            self.moving_object_name = next(iter(self.objects_dict.keys()))

        if self.moving_object_name not in self.objects_dict:
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_base_qpos = None
            return

        moving_object = self.objects_dict[self.moving_object_name]
        if not getattr(moving_object, "joints", None):
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_base_qpos = None
            return

        self._motion_joint_name = moving_object.joints[-1]
        joint_id = self.sim.model.joint_name2id(self._motion_joint_name)
        qvel_addr = int(self.sim.model.jnt_dofadr[joint_id])
        self._motion_qvel_slice = slice(qvel_addr, qvel_addr + 6)
        self._motion_base_qpos = np.array(
            self.sim.data.get_joint_qpos(self._motion_joint_name), dtype=np.float64
        ).copy()

    def _apply_position_drag(self, delta):
        if not self.motion_enabled:
            return
        if self._motion_joint_name is None or self._motion_base_qpos is None:
            return

        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_to_idx.get(self.motion_axis.lower(), 0)
        new_qpos = np.array(self.sim.data.get_joint_qpos(self._motion_joint_name), dtype=np.float64).copy()
        new_qpos[axis_idx] = float(new_qpos[axis_idx] + delta)
        self.sim.data.set_joint_qpos(self._motion_joint_name, new_qpos)
        self.sim.forward()

    def _get_drive_delta_for_step(self, drive_step_idx: int) -> float:
        if self.motion_drive_steps <= 0:
            return 0.0

        if not self.motion_return_enabled:
            if self.motion_drive_steps == 1:
                return self.motion_end_speed
            alpha = float(drive_step_idx) / float(self.motion_drive_steps - 1)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            return (1.0 - alpha) * self.motion_drive_speed + alpha * self.motion_end_speed

        outbound_steps = max(1, self.motion_drive_steps // 2)
        return_steps = max(1, self.motion_drive_steps - outbound_steps)

        if drive_step_idx < outbound_steps:
            if outbound_steps == 1:
                return self.motion_drive_speed
            alpha = float(drive_step_idx) / float(outbound_steps - 1)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            return (1.0 - alpha) * self.motion_drive_speed + alpha * self.motion_end_speed

        return_idx = drive_step_idx - outbound_steps
        if return_steps == 1:
            return -self.motion_drive_speed
        alpha = float(return_idx) / float(return_steps - 1)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return -((1.0 - alpha) * self.motion_drive_speed + alpha * self.motion_end_speed)

    def _post_process(self):
        super()._post_process()
        if not self.motion_enabled or self._velocity_injected:
            return
        self._post_reset_step_idx += 1
        if self._post_reset_step_idx < self.motion_settle_steps:
            return

        if self._drive_step_idx < self.motion_drive_steps:
            current_delta = self._get_drive_delta_for_step(self._drive_step_idx)
            self._apply_position_drag(current_delta)
            self._drive_step_idx += 1
        else:
            self._velocity_injected = True

    def get_dynamic_object_position(self):
        if self.moving_object_name is None or self.moving_object_name not in self.objects_dict:
            return None
        moving_object = self.objects_dict[self.moving_object_name]
        root_body = getattr(moving_object, "root_body", None)
        if root_body is None:
            return None
        try:
            body_id = self.sim.model.body_name2id(root_body)
        except Exception:
            return None
        return np.array(self.sim.data.body_xpos[body_id], dtype=np.float64).copy()
