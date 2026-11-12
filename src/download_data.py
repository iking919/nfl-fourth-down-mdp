
"""
Download the 2024 NFL play-by-play dataset from nflverse.
"""

from pathlib import Path
from urllib.request import Request, urlopen
import shutil


# Change this if we want to download a different season.
SEASON = 2024

URL = (
    f"https://github.com/nflverse/nflverse-data/"
    f"releases/download/pbp/play_by_play_{SEASON}.csv"
)

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / f"play_by_play_{SEASON}.csv"


def download_pbp():
    # Make sure the raw data directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid downloading the same file again by accident.
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists: {OUTPUT_FILE}")
        print("Delete it manually if you want to download it again.")
        return

    print(f"Downloading {SEASON} NFL play-by-play data...")
    print(f"Source: {URL}")

    # Include a user agent in the request so GitHub accepts it.
    request = Request(
        URL,
        headers={"User-Agent": "nfl-fourth-down-mdp"}
    )

    try:
        # Stream the download directly to disk instead of loading
        # the entire CSV into memory.
        with urlopen(request, timeout=120) as response:
            with open(OUTPUT_FILE, "wb") as output:
                shutil.copyfileobj(response, output)

        size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)

        print("\nDownload complete!")
        print(f"Saved to: {OUTPUT_FILE}")
        print(f"File size: {size_mb:.2f} MB")

    except Exception as error:
        # Remove an incomplete file if the download fails.
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

        print(f"Download failed: {error}")
        raise


if __name__ == "__main__":
    download_pbp()
