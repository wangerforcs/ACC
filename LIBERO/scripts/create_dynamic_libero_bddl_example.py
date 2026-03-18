"""
Create a dynamic-task BDDL by copying an existing LIBERO BDDL and changing the
problem name so it maps to Libero_Dynamic_Floor_Manipulation.

This is the minimum plumbing needed because LIBERO chooses the environment
class from the BDDL problem name.
"""

from pathlib import Path
import argparse


SOURCE_DEFAULT = (
    "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_object/"
    "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl"
)
TARGET_DIR_DEFAULT = (
    "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_dynamic_examples"
)
TARGET_PROBLEM_NAME = "LIBERO_Dynamic_Floor_Manipulation"


def patch_problem_name(bddl_text: str, new_problem_name: str) -> str:
    lines = bddl_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("(define (problem "):
            lines[i] = f"(define (problem {new_problem_name})"
            break
    else:
        raise ValueError("Could not find '(define (problem ...)' in BDDL file.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bddl", type=str, default=SOURCE_DEFAULT)
    parser.add_argument("--target-dir", type=str, default=TARGET_DIR_DEFAULT)
    parser.add_argument(
        "--target-name",
        type=str,
        default="pick_up_the_orange_juice_and_place_it_in_the_moving_basket.bddl",
    )
    args = parser.parse_args()

    source_path = Path(args.source_bddl)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / args.target_name

    text = source_path.read_text()
    patched = patch_problem_name(text, TARGET_PROBLEM_NAME)
    target_path.write_text(patched)

    print(f"Wrote dynamic example BDDL to: {target_path}")
    print("Use it with:")
    print(
        "  from libero.libero.envs import OffScreenRenderEnv\n"
        f"  env = OffScreenRenderEnv(bddl_file_name='{target_path}', "
        "robots=['Panda'], moving_object_name='orange_juice_1', "
        "motion_axis='x', motion_initial_speed=0.05)"
    )


if __name__ == "__main__":
    main()
