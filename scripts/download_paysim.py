"""CLI for downloading the public synthetic PaySim dataset."""

import argparse

from src.data.ingest import DEFAULT_URL, download_paysim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/raw/paysim.csv")
    args = parser.parse_args()
    print(f"Wrote {download_paysim(args.url, args.output)}")


if __name__ == "__main__":
    main()