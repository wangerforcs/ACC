import numpy as np

from libero.libero.envs.bddl_base_domain import register_problem
from libero.libero.envs.problems.libero_floor_manipulation import (
    Libero_Floor_Manipulation,
)


@register_problem
class Libero_Dynamic_Floor_Manipulation(Libero_Floor_Manipulation):
    """
    Minimal example of a dynamic LIBERO environment.

    This environment reuses the standard floor-manipulation logic but adds a
    post-reset motion policy for one movable object. After a short settle
    period, the object slides in the table plane while keeping its initialized
    pose fixed in z / orientation. The scripted path can be a straight line, a
    diagonal line, or a simple right-angle path. Once the object is grasped,
    scripted motion support is disabled immediately.
    """

    def __init__(
        self,
        bddl_file_name,
        *args,
        moving_object_name=None,
        motion_axis="x",
        motion_path="line",
        motion_secondary_axis=None,
        motion_turn_direction="left",
        motion_turn_step=None,
        motion_turn_cycles=1,
        motion_initial_speed=None,
        motion_surface_friction_scale=None,
        motion_settle_steps=15,
        motion_drive_steps=30,
        motion_drive_speed=0.01,
        motion_secondary_speed=None,
        motion_end_speed=0.0,
        motion_return_enabled=False,
        motion_max_angular_speed=0.0,
        motion_friction_scale=0.15,
        motion_friction_restore_step=None,
        motion_object_resistance_scale=None,
        motion_post_restore_velocity_decay=0.97,
        motion_amplitude=0.15,
        motion_frequency_hz=0.01,
        motion_enabled=True,
        **kwargs,
    ):
        self.moving_object_name = moving_object_name
        self.motion_axis = motion_axis
        self.motion_path = str(motion_path).lower()
        self.motion_secondary_axis = motion_secondary_axis
        self.motion_turn_direction = str(motion_turn_direction).lower()
        self.motion_turn_step = None if motion_turn_step is None else max(0, int(motion_turn_step))
        self.motion_turn_cycles = int(motion_turn_cycles)
        # Keep backward compatibility with older configs that used amplitude /
        # frequency driven sinusoidal motion. If explicit initial speed is not
        # provided, derive a small velocity scale from the old parameters.
        if motion_initial_speed is None:
            motion_initial_speed = 2.0 * np.pi * float(motion_frequency_hz) * float(motion_amplitude)
        self.motion_initial_speed = float(motion_initial_speed)
        self.motion_surface_friction_scale = (
            None if motion_surface_friction_scale is None else float(motion_surface_friction_scale)
        )
        self.motion_settle_steps = max(0, int(motion_settle_steps))
        self.motion_drive_steps = max(0, int(motion_drive_steps))
        self.motion_drive_speed = float(motion_drive_speed)
        if motion_secondary_speed is None:
            motion_secondary_speed = motion_drive_speed
        self.motion_secondary_speed = float(motion_secondary_speed)
        self.motion_end_speed = float(motion_end_speed)
        self.motion_return_enabled = bool(motion_return_enabled)
        self.motion_max_angular_speed = float(motion_max_angular_speed)
        self.motion_friction_scale = float(motion_friction_scale)
        self.motion_friction_restore_step = (
            None if motion_friction_restore_step is None else max(0, int(motion_friction_restore_step))
        )
        self.motion_object_resistance_scale = (
            None if motion_object_resistance_scale is None else float(motion_object_resistance_scale)
        )
        self.motion_post_restore_velocity_decay = float(motion_post_restore_velocity_decay)
        self.motion_amplitude = float(motion_amplitude)
        self.motion_frequency_hz = float(motion_frequency_hz)
        self.motion_enabled = bool(motion_enabled)

        self._motion_joint_name = None
        self._motion_qvel_slice = None
        self._motion_dof_slice = None
        self._motion_base_qpos = None
        self._motion_body_id = None
        self._motion_geom_ids = []
        self._original_geom_friction = {}
        self._support_geom_ids = []
        self._original_support_geom_friction = {}
        self._original_dof_damping = None
        self._original_dof_frictionloss = None
        self._friction_restored = False
        self._scripted_motion_disabled = False
        self._motion_axis_idx = 0
        self._motion_secondary_axis_idx = 1
        self._post_reset_step_idx = 0
        self._drive_step_idx = 0
        self._velocity_injected = False
        self._motion_stopped_by_grasp = False
        self._slide_velocity_primary = 0.0
        self._slide_velocity_secondary = 0.0
        self._return_direction = 1.0
        super().__init__(bddl_file_name, *args, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        self._cache_motion_reference()
        self._cache_support_surface_reference()
        self._apply_support_surface_friction_override()
        self._apply_motion_friction_override()
        self._post_reset_step_idx = 0
        self._drive_step_idx = 0
        self._velocity_injected = False
        self._motion_stopped_by_grasp = False
        self._slide_velocity_primary = 0.0
        self._slide_velocity_secondary = 0.0
        self._return_direction = 1.0
        self._friction_restored = False
        self._scripted_motion_disabled = False
        self._apply_initial_pose_override()

    def _cache_motion_reference(self):
        if not self.motion_enabled:
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_dof_slice = None
            self._motion_base_qpos = None
            self._motion_body_id = None
            return

        if self.moving_object_name is None:
            if len(self.objects_dict) == 0:
                self._motion_joint_name = None
                self._motion_qvel_slice = None
                self._motion_dof_slice = None
                self._motion_base_qpos = None
                self._motion_body_id = None
                return
            self.moving_object_name = next(iter(self.objects_dict.keys()))

        if self.moving_object_name not in self.objects_dict:
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_dof_slice = None
            self._motion_base_qpos = None
            self._motion_body_id = None
            return

        moving_object = self.objects_dict[self.moving_object_name]
        if not getattr(moving_object, "joints", None):
            self._motion_joint_name = None
            self._motion_qvel_slice = None
            self._motion_dof_slice = None
            self._motion_base_qpos = None
            self._motion_body_id = None
            return

        self._motion_joint_name = moving_object.joints[-1]
        joint_id = self.sim.model.joint_name2id(self._motion_joint_name)
        qvel_addr = int(self.sim.model.jnt_dofadr[joint_id])
        self._motion_qvel_slice = slice(qvel_addr, qvel_addr + 6)
        self._motion_dof_slice = slice(qvel_addr, qvel_addr + 6)
        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        self._motion_axis_idx = axis_to_idx.get(self.motion_axis.lower(), 0)
        secondary_axis = self.motion_secondary_axis or ("x" if self._motion_axis_idx != 0 else "y")
        self._motion_secondary_axis_idx = axis_to_idx.get(str(secondary_axis).lower(), 1)
        if self._motion_secondary_axis_idx == self._motion_axis_idx:
            self._motion_secondary_axis_idx = 1 if self._motion_axis_idx != 1 else 0
        self._motion_base_qpos = np.array(
            self.sim.data.get_joint_qpos(self._motion_joint_name), dtype=np.float64
        ).copy()
        root_body = getattr(moving_object, "root_body", None)
        self._motion_body_id = self.sim.model.body_name2id(root_body) if root_body is not None else None
        self._motion_geom_ids = []
        self._original_geom_friction = {}
        self._original_dof_damping = np.array(self.sim.model.dof_damping[self._motion_dof_slice], dtype=np.float64).copy()
        self._original_dof_frictionloss = np.array(
            self.sim.model.dof_frictionloss[self._motion_dof_slice], dtype=np.float64
        ).copy()
        if self._motion_body_id is not None:
            body_geom_start = int(self.sim.model.body_geomadr[self._motion_body_id])
            body_geom_count = int(self.sim.model.body_geomnum[self._motion_body_id])
            self._motion_geom_ids = list(range(body_geom_start, body_geom_start + body_geom_count))
            for geom_id in self._motion_geom_ids:
                self._original_geom_friction[geom_id] = np.array(
                    self.sim.model.geom_friction[geom_id], dtype=np.float64
                ).copy()

    def _cache_support_surface_reference(self):
        self._support_geom_ids = []
        self._original_support_geom_friction = {}
        if self.motion_surface_friction_scale is None:
            return
        for geom_id in range(int(self.sim.model.ngeom)):
            body_id = int(self.sim.model.geom_bodyid[geom_id])
            body_name = self.sim.model.body_id2name(body_id) or ""
            geom_name = self.sim.model.geom_id2name(geom_id) or ""
            lower = f"{body_name} {geom_name}".lower()
            if "floor" in lower or "table" in lower:
                self._support_geom_ids.append(geom_id)
                self._original_support_geom_friction[geom_id] = np.array(
                    self.sim.model.geom_friction[geom_id], dtype=np.float64
                ).copy()

    def _stop_motion(self):
        if self._motion_qvel_slice is None:
            return
        self.sim.data.qvel[self._motion_qvel_slice] = 0.0

    def _apply_post_restore_velocity_decay(self):
        if self._motion_qvel_slice is None:
            return
        qvel = self.sim.data.qvel[self._motion_qvel_slice].copy()
        decay = float(np.clip(self.motion_post_restore_velocity_decay, 0.0, 1.0))
        qvel[:3] *= decay
        if np.linalg.norm(qvel[:2]) < 1e-4:
            qvel[:2] = 0.0
        if abs(qvel[2]) < 1e-4:
            qvel[2] = 0.0
        qvel[3:] = 0.0
        self.sim.data.qvel[self._motion_qvel_slice] = qvel

    def _transfer_scripted_velocity_to_physics(self):
        if self._motion_qvel_slice is None:
            return
        qvel = self.sim.data.qvel[self._motion_qvel_slice].copy()
        qvel[:] = 0.0
        control_freq = float(getattr(self, "control_freq", 20.0))
        velocity_scale = max(control_freq, 1.0)

        if self.motion_path == "diagonal":
            primary_velocity = float(self._slide_velocity_primary) * velocity_scale
            diagonal_sign = -1.0 if self.motion_turn_direction == "right" else 1.0
            secondary_velocity = diagonal_sign * float(self._slide_velocity_secondary) * velocity_scale
            qvel[self._motion_axis_idx] = primary_velocity
            qvel[self._motion_secondary_axis_idx] = secondary_velocity
        elif self.motion_path == "right_angle":
            segment_len = max(
                1,
                self.motion_turn_step if self.motion_turn_step is not None else max(1, self.motion_drive_steps // 2),
            )
            secondary_sign = -1.0 if self.motion_turn_direction == "right" else 1.0
            if self.motion_turn_cycles == 1:
                if self._drive_step_idx >= segment_len:
                    qvel[self._motion_secondary_axis_idx] = (
                        secondary_sign * float(self._slide_velocity_secondary) * velocity_scale
                    )
                else:
                    qvel[self._motion_axis_idx] = float(self._slide_velocity_primary) * velocity_scale
            else:
                segment_idx = self._drive_step_idx // segment_len
                phase = segment_idx % 4
                if phase == 0:
                    qvel[self._motion_axis_idx] = abs(float(self._slide_velocity_primary)) * velocity_scale
                elif phase == 1:
                    qvel[self._motion_secondary_axis_idx] = (
                        secondary_sign * abs(float(self._slide_velocity_secondary)) * velocity_scale
                    )
                elif phase == 2:
                    qvel[self._motion_axis_idx] = -abs(float(self._slide_velocity_primary)) * velocity_scale
                else:
                    qvel[self._motion_secondary_axis_idx] = (
                        -secondary_sign * abs(float(self._slide_velocity_secondary)) * velocity_scale
                    )
        else:
            qvel[self._motion_axis_idx] = float(self._slide_velocity_primary) * velocity_scale

        self.sim.data.qvel[self._motion_qvel_slice] = qvel

    def _inject_initial_velocity_to_physics(self):
        if self._motion_qvel_slice is None:
            return
        qvel = self.sim.data.qvel[self._motion_qvel_slice].copy()
        qvel[:] = 0.0
        control_freq = float(getattr(self, "control_freq", 20.0))
        velocity_scale = max(control_freq, 1.0)
        direction = float(getattr(self, "_return_direction", 1.0))
        primary_velocity = direction * float(self.motion_initial_speed) * velocity_scale

        if self.motion_path == "diagonal":
            diagonal_sign = -1.0 if self.motion_turn_direction == "right" else 1.0
            secondary_mag = abs(float(self.motion_secondary_speed))
            if secondary_mag < 1e-8:
                secondary_mag = abs(float(self.motion_initial_speed))
            qvel[self._motion_axis_idx] = primary_velocity
            qvel[self._motion_secondary_axis_idx] = (
                direction * diagonal_sign * secondary_mag * velocity_scale
            )
        else:
            qvel[self._motion_axis_idx] = primary_velocity

        self.sim.data.qvel[self._motion_qvel_slice] = qvel

    def _maybe_reverse_motion(self):
        if not self.motion_return_enabled:
            return False
        if self._motion_qvel_slice is None:
            return False
        qvel = np.array(self.sim.data.qvel[self._motion_qvel_slice], dtype=np.float64)
        planar_speed = float(np.linalg.norm(qvel[:2]))
        reverse_threshold = max(1e-4, abs(float(self.motion_end_speed)))
        if planar_speed > reverse_threshold:
            return False
        self._return_direction *= -1.0
        self._inject_initial_velocity_to_physics()
        return True

    def _apply_motion_friction_override(self):
        if self._motion_body_id is None:
            return
        if self.motion_friction_scale <= 0:
            return
        for geom_id in self._motion_geom_ids:
            base_friction = self._original_geom_friction.get(geom_id)
            if base_friction is None:
                base_friction = np.array(self.sim.model.geom_friction[geom_id], dtype=np.float64)
                self._original_geom_friction[geom_id] = base_friction.copy()
            friction = np.array(base_friction, dtype=np.float64)
            friction *= self.motion_friction_scale
            # Keep a tiny positive floor to avoid pathological zero-friction contact.
            friction = np.maximum(friction, np.array([1e-4, 1e-4, 1e-4], dtype=np.float64))
            self.sim.model.geom_friction[geom_id] = friction
        self._friction_restored = False

    def _apply_support_surface_friction_override(self):
        if self.motion_surface_friction_scale is None:
            return
        for geom_id in self._support_geom_ids:
            base_friction = self._original_support_geom_friction.get(geom_id)
            if base_friction is None:
                continue
            friction = np.array(base_friction, dtype=np.float64)
            friction *= self.motion_surface_friction_scale
            friction = np.maximum(friction, np.array([0.0, 0.0, 0.0], dtype=np.float64))
            self.sim.model.geom_friction[geom_id] = friction

    def _apply_object_resistance_override(self):
        if self._motion_body_id is None:
            return
        if self.motion_object_resistance_scale is None:
            return
        # Only scale the moving object's internal resistance here. Contact
        # friction against the carried object is controlled separately via
        # `motion_friction_scale`, otherwise a low resistance setting makes the
        # plate too slippery to carry the object after placement.
        if self._motion_dof_slice is not None and self._original_dof_damping is not None:
            damping = np.array(self._original_dof_damping, dtype=np.float64) * self.motion_object_resistance_scale
            frictionloss = (
                np.array(self._original_dof_frictionloss, dtype=np.float64) * self.motion_object_resistance_scale
            )
            self.sim.model.dof_damping[self._motion_dof_slice] = damping
            self.sim.model.dof_frictionloss[self._motion_dof_slice] = frictionloss

    def _restore_motion_friction_if_needed(self):
        if self._friction_restored:
            return
        if self.motion_friction_restore_step is None:
            return
        if self._post_reset_step_idx < self.motion_friction_restore_step:
            return
        if self.motion_object_resistance_scale is not None:
            self._apply_object_resistance_override()
        else:
            for geom_id in self._motion_geom_ids:
                base_friction = self._original_geom_friction.get(geom_id)
                if base_friction is not None:
                    self.sim.model.geom_friction[geom_id] = np.array(base_friction, dtype=np.float64)
        self._friction_restored = True
        # At the handoff step, stop scripted qpos driving and let physics
        # carry the object with the current planar velocity. Friction can
        # either be restored here or intentionally kept low.
        if not self._scripted_motion_disabled:
            self._transfer_scripted_velocity_to_physics()
            self._scripted_motion_disabled = True

    def _apply_initial_pose_override(self):
        if self._motion_joint_name is None:
            return
        if self.moving_object_name != "orange_juice_1":
            return
        new_qpos = np.array(self.sim.data.get_joint_qpos(self._motion_joint_name), dtype=np.float64).copy()
        # new_qpos[3:7] = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64) #横着
        new_qpos[3:7] = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float64) #竖着
        self.sim.data.set_joint_qpos(self._motion_joint_name, new_qpos)
        self.sim.forward()

    def _apply_planar_slide(self, drive_step_idx: int):
        if self._motion_joint_name is None:
            return
        if drive_step_idx == 0 and abs(self._slide_velocity_primary) < 1e-8:
            start_speed = float(self.motion_drive_speed)
            if abs(start_speed) < 1e-8:
                start_speed = float(self.motion_initial_speed)
            self._slide_velocity_primary = start_speed
            self._slide_velocity_secondary = float(self.motion_secondary_speed)

        drive_limit = self.motion_drive_steps if self.motion_drive_steps > 0 else 12
        turn_step = self.motion_turn_step if self.motion_turn_step is not None else max(1, drive_limit // 2)
        if self.motion_path == "diagonal":
            if abs(self._slide_velocity_primary) < 1e-8 and abs(self._slide_velocity_secondary) < 1e-8:
                self._velocity_injected = True
                self._stop_motion()
                return
        elif self.motion_path == "right_angle":
            if self.motion_turn_cycles == 1:
                pass
            else:
                segment_len = max(1, turn_step)
                segment_idx = drive_step_idx // segment_len
                if self.motion_turn_cycles > 0 and segment_idx >= 4 * self.motion_turn_cycles:
                    self._velocity_injected = True
                    self._stop_motion()
                    return
        elif abs(self._slide_velocity_primary) < 1e-8:
            self._velocity_injected = True
            self._stop_motion()
            return

        # Drive the object by velocity rather than teleporting qpos. This lets
        # MuJoCo contact dynamics carry other objects (for example cheese on a
        # moving plate) instead of the support object sliding underneath them.
        self._transfer_scripted_velocity_to_physics()

        decay = 0.96 if drive_step_idx < drive_limit else 0.90
        end_speed = float(self.motion_end_speed)
        if self.motion_path == "diagonal":
            self._slide_velocity_primary = max(abs(self.motion_drive_speed), 1e-4)
            self._slide_velocity_secondary = max(abs(self.motion_secondary_speed), 1e-4)
        elif self.motion_path == "right_angle" and self.motion_turn_cycles != 1:
            # For cyclic square motion, keep a stable scripted speed instead of
            # decaying toward zero. Otherwise the path visually slows down and
            # appears to stop after a while.
            self._slide_velocity_primary = max(abs(self.motion_drive_speed), 1e-4)
            self._slide_velocity_secondary = max(abs(self.motion_secondary_speed), 1e-4)
        else:
            if self.motion_path == "right_angle" and drive_step_idx >= turn_step:
                self._slide_velocity_secondary *= decay
                if abs(self._slide_velocity_secondary) < max(abs(end_speed), 1e-4):
                    self._slide_velocity_secondary = 0.0
            else:
                self._slide_velocity_primary *= decay
                if abs(self._slide_velocity_primary) < max(abs(end_speed), 1e-4):
                    self._slide_velocity_primary = 0.0

            if (
                abs(self._slide_velocity_primary) < max(abs(end_speed), 1e-4)
                and abs(self._slide_velocity_secondary) < max(abs(end_speed), 1e-4)
            ):
                self._velocity_injected = True

    def _is_motion_object_grasped(self) -> bool:
        if self.moving_object_name is None or self.moving_object_name not in self.objects_dict:
            return False
        if not getattr(self, "robots", None):
            return False
        moving_object = self.objects_dict[self.moving_object_name]
        try:
            return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=moving_object))
        except Exception:
            contact_geoms = getattr(moving_object, "contact_geoms", None)
            if not contact_geoms:
                return False
            try:
                return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=contact_geoms))
            except Exception:
                return False

    def _post_process(self):
        super()._post_process()
        if not self.motion_enabled:
            return
        self._post_reset_step_idx += 1
        self._restore_motion_friction_if_needed()
        if self._motion_stopped_by_grasp:
            return

        if self._is_motion_object_grasped():
            self._stop_motion()
            self._motion_stopped_by_grasp = True
            self._velocity_injected = True
            return

        if self._velocity_injected:
            if self._maybe_reverse_motion():
                self._velocity_injected = False
            return

        if self._scripted_motion_disabled:
            return

        if self._post_reset_step_idx < self.motion_settle_steps:
            return

        initial_velocity_only = (
            abs(float(self.motion_initial_speed)) > 1e-8
            and abs(float(self.motion_drive_speed)) < 1e-8
            and abs(float(self.motion_secondary_speed)) < 1e-8
        )
        if initial_velocity_only:
            if not self._friction_restored:
                if self.motion_object_resistance_scale is not None:
                    self._apply_object_resistance_override()
                else:
                    for geom_id in self._motion_geom_ids:
                        base_friction = self._original_geom_friction.get(geom_id)
                        if base_friction is not None:
                            self.sim.model.geom_friction[geom_id] = np.array(base_friction, dtype=np.float64)
                self._friction_restored = True
            self._inject_initial_velocity_to_physics()
            self._velocity_injected = True
            self._scripted_motion_disabled = True
            return

        self._apply_planar_slide(self._drive_step_idx)
        self._drive_step_idx += 1

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

    def is_dynamic_object_grasped(self):
        return self._is_motion_object_grasped()
