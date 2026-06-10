import tarfile
import pathlib
import argparse
from src.utils.logging_utils import get_logger

logger = get_logger("extract")


def extract_caravan(archive_path: str, dest: str = "data/raw") -> None:
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path} -> {dest}")
    with tarfile.open(archive_path, "r:xz") as tar:
        members = tar.getmembers()
        logger.info(f"Archive contains {len(members)} entries")
        tar.extractall(dest)
    logger.info(f"Extraction complete. Files in {dest}:")
    for p in sorted(dest.rglob("*"))[:20]:
        logger.info(f"  {p.relative_to(dest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="Caravan-csv.tar.xz")
    parser.add_argument("--dest", default="data/raw")
    args = parser.parse_args()
    extract_caravan(args.archive, args.dest)
