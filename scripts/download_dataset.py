import argparse
import os

from roboflow import Roboflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dataset Roboflow In Format YOLO26"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset Version (Default: 1)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY") or input(
        "Insert Roboflow API key (Display Text): "
    )
    if not api_key:
        raise SystemExit("No API key inserted")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("khachonyod-komin").project("thai-char-detection")
    version = project.version(args.version)
    dataset = version.download("yolo26")
    print(f"Download Asset Success: {dataset.location}")


if __name__ == "__main__":
    main()
