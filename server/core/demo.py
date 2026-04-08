from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.core.env.execution_desk_env import heuristic_policy, run_demo


def main() -> None:
    run_demo(policy=heuristic_policy)


if __name__ == "__main__":
    main()
