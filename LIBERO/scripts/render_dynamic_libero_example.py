import argparse
import json
from pathlib import Path

import numpy as np
import cv2

try:
    import imageio
except ImportError:
    imageio = None

from libero.libero.envs import OffScreenRenderEnv


DEFAULT_BDDL = (
    "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_dynamic_examples/"
    "test_dynamic_example.bddl"
)
DEFAULT_OUTPUT = (
    "/home/wangzb/vlash/LIBERO/tmp/dynamic_libero_example_agentview.mp4"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, default=DEFAULT_BDDL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--moving-object-name", type=str, default="orange_juice_1")
    parser.add_argument("--motion-axis", type=str, default="y")
    parser.add_argument("--motion-path", type=str, default="line")
    parser.add_argument("--motion-secondary-axis", type=str, default=None)
    parser.add_argument("--motion-turn-direction", type=str, default="left")
    parser.add_argument("--motion-turn-step", type=int, default=None)
    parser.add_argument("--motion-turn-cycles", type=int, default=1)
    parser.add_argument("--motion-initial-speed", type=float, default=None)
    parser.add_argument("--motion-surface-friction-scale", type=float, default=None)
    parser.add_argument("--motion-settle-steps", type=int, default=15)
    parser.add_argument("--motion-drive-steps", type=int, default=12)
    parser.add_argument("--motion-drive-speed", type=float, default=0.01)
    parser.add_argument("--motion-secondary-speed", type=float, default=None)
    parser.add_argument("--motion-end-speed", type=float, default=0.0)
    parser.add_argument("--motion-friction-scale", type=float, default=0.15)
    parser.add_argument("--motion-friction-restore-step", type=int, default=None)
    parser.add_argument("--motion-object-resistance-scale", type=float, default=None)
    parser.add_argument("--motion-post-restore-velocity-decay", type=float, default=0.97)
    parser.add_argument("--motion-return-enabled", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--motion-max-angular-speed", type=float, default=0.0)
    parser.add_argument("--motion-amplitude", type=float, default=0.15)
    parser.add_argument("--motion-frequency-hz", type=float, default=0.01)
    parser.add_argument("--print-position-every", type=int, default=1)
    parser.add_argument("--position-json", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        robots=[args.robot],
        camera_names=[args.camera_name],
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        moving_object_name=args.moving_object_name,
        motion_axis=args.motion_axis,
        motion_path=args.motion_path,
        motion_secondary_axis=args.motion_secondary_axis,
        motion_turn_direction=args.motion_turn_direction,
        motion_turn_step=args.motion_turn_step,
        motion_turn_cycles=args.motion_turn_cycles,
        motion_initial_speed=args.motion_initial_speed,
        motion_surface_friction_scale=args.motion_surface_friction_scale,
        motion_settle_steps=args.motion_settle_steps,
        motion_drive_steps=args.motion_drive_steps,
        motion_drive_speed=args.motion_drive_speed,
        motion_secondary_speed=args.motion_secondary_speed,
        motion_end_speed=args.motion_end_speed,
        motion_friction_scale=args.motion_friction_scale,
        motion_friction_restore_step=args.motion_friction_restore_step,
        motion_object_resistance_scale=args.motion_object_resistance_scale,
        motion_post_restore_velocity_decay=args.motion_post_restore_velocity_decay,
        motion_return_enabled=args.motion_return_enabled,
        motion_max_angular_speed=args.motion_max_angular_speed,
        motion_amplitude=args.motion_amplitude,
        motion_frequency_hz=args.motion_frequency_hz,
    )

    try:
        obs = env.reset()
        frame_key = f"{args.camera_name}_image"
        frames = [obs[frame_key][::-1].copy()]
        position_trace = []

        def record_position(step_idx: int):
            pos = None
            grasped = None
            if hasattr(env.env, "get_dynamic_object_position"):
                pos = env.env.get_dynamic_object_position()
            if hasattr(env.env, "is_dynamic_object_grasped"):
                grasped = bool(env.env.is_dynamic_object_grasped())
            if pos is None:
                return
            pos_list = [float(v) for v in pos]
            record = {"step": step_idx, "position": pos_list}
            if grasped is not None:
                record["grasped"] = grasped
            position_trace.append(record)
            if args.print_position_every > 0 and step_idx % args.print_position_every == 0:
                if grasped is None:
                    print(f"step={step_idx} pos={pos_list}")
                else:
                    print(f"step={step_idx} pos={pos_list} grasped={grasped}")

        action_dim = env.env.action_dim
        zero_action = np.zeros(action_dim, dtype=np.float32)
        record_position(0)

        for step_idx in range(1, args.steps + 1):
            obs, _, done, _ = env.step(zero_action)
            frames.append(obs[frame_key][::-1].copy())
            record_position(step_idx)
            if done:
                break

        if imageio is not None:
            with imageio.get_writer(output_path, fps=args.fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
        else:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (w, h),
            )
            for frame in frames:
                writer.write(frame[:, :, ::-1])
            writer.release()

        print(f"Saved video to: {output_path}")
        print(f"Frames written: {len(frames)}")
        print(f"Dynamic object: {args.moving_object_name}")
        if args.position_json:
            position_path = Path(args.position_json)
            position_path.parent.mkdir(parents=True, exist_ok=True)
            position_path.write_text(json.dumps(position_trace, indent=2))
            print(f"Saved position trace to: {position_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
