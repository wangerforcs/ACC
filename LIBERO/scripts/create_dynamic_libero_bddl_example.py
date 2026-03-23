"""
Create a clean dynamic-task BDDL with only the task-essential objects.

By default this emits the original minimal orange-juice-into-basket scene.
Alternative dynamic-task BDDL files can live alongside it in the same folder.
"""

from pathlib import Path
import argparse


TARGET_DIR_DEFAULT = (
    "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_dynamic_examples"
)
TARGET_PROBLEM_NAME = "LIBERO_Dynamic_Floor_Manipulation"
MINIMAL_DYNAMIC_BDDL = f"""(define (problem {TARGET_PROBLEM_NAME})
  (:domain robosuite)
  (:language Pick the orange juice and place it in the basket)
    (:regions
      (bin_region
          (:target floor)
          (:ranges (
              (-0.01 0.25 0.01 0.27)
            )
          )
      )
      (target_object_region
          (:target floor)
          (:ranges (
              (0.025 -0.125 0.07500000000000001 -0.07500000000000001)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    floor - floor
  )

  (:objects
    orange_juice_1 - orange_juice
    basket_1 - basket
  )

  (:obj_of_interest
    orange_juice_1
    basket_1
  )

  (:init
    (On orange_juice_1 floor_target_object_region)
    (On basket_1 floor_bin_region)
  )

  (:goal
    (And (In orange_juice_1 basket_1_contain_region))
  )

)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default=TARGET_DIR_DEFAULT)
    parser.add_argument(
        "--target-name",
        type=str,
        default="test_dynamic_example.bddl",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / args.target_name

    target_path.write_text(MINIMAL_DYNAMIC_BDDL)

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
