"""Print a trajectory from deterministic synthetic Stage 9 demo data."""

import argparse
import json

from backend.trajectory.demo import build_demo_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plate", nargs="?", default="KA02MN1826")
    args = parser.parse_args()
    print(json.dumps(build_demo_service().get_trajectory(args.plate).to_dict(), indent=2))


if __name__ == "__main__":
    main()
