#!/usr/bin/env python3
"""
Test a custom dynamic LIBERO environment by directly loading a dynamic BDDL file.

This script:
- creates OffScreenRenderEnv from a dynamic BDDL
- steps the environment with zero actions
- saves a rendered video
- exports tracked object positions as json
"""

import json
import os
from pathlib import Path
import argparse

import cv2
import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

from libero.libero.envs import OffScreenRenderEnv


DEFAULT_BDDL = (
    "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_dynamic_examples/"
    "test_dynamic_example.bddl"
)
DEFAULT_OUTDIR = "/home/wangzb/vlash/adaptive/dynamic_libero_results"


def save_video(frames, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(frame[:, :, ::-1])
    writer.release()


def get_object_position(env, object_name: str):
    if object_name not in env.env.obj_body_id:
        return None
    pos = env.sim.data.body_xpos[env.env.obj_body_id[object_name]]
    return [float(x) for x in pos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, default=DEFAULT_BDDL)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--moving-object-name", type=str, default="orange_juice_1")
    parser.add_argument("--motion-axis", type=str, default="x")
    parser.add_argument("--motion-amplitude", type=float, default=0.05)
    parser.add_argument("--motion-frequency-hz", type=float, default=0.15)
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        robots=[args.robot],
        camera_names=[args.camera_name],
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        moving_object_name=args.moving_object_name,
        motion_axis=args.motion_axis,
        motion_amplitude=args.motion_amplitude,
        motion_frequency_hz=args.motion_frequency_hz,
    )

    frame_key = f"{args.camera_name}_image"
    video_path = outdir / "dynamic_env_render.mp4"
    metadata_path = outdir / "dynamic_env_metadata.json"

    frames = []
    positions = []

    try:
        obs = env.reset()
        frames.append(obs[frame_key][::-1].copy())
        positions.append(
            {
                "step": 0,
                "moving_object_name": args.moving_object_name,
                "position": get_object_position(env, args.moving_object_name),
            }
        )

        action_dim = env.env.action_dim
        zero_action = np.zeros(action_dim, dtype=np.float32)

        for step_idx in range(1, args.steps + 1):
            obs, reward, done, info = env.step(zero_action)
            frames.append(obs[frame_key][::-1].copy())
            positions.append(
                {
                    "step": step_idx,
                    "reward": float(reward),
                    "done": bool(done),
                    "moving_object_name": args.moving_object_name,
                    "position": get_object_position(env, args.moving_object_name),
                }
            )
            if done:
                break

        save_video(frames, video_path, args.fps)

        metadata = {
            "bddl_file": args.bddl_file,
            "moving_object_name": args.moving_object_name,
            "motion_axis": args.motion_axis,
            "motion_amplitude": args.motion_amplitude,
            "motion_frequency_hz": args.motion_frequency_hz,
            "num_frames": len(frames),
            "positions": positions,
            "video_path": str(video_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        print(f"Saved video to: {video_path}")
        print(f"Saved metadata to: {metadata_path}")
        if positions and positions[0]['position'] is not None and positions[-1]['position'] is not None:
            print(f"Start position: {positions[0]['position']}")
            print(f"End position:   {positions[-1]['position']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
