"""Download the public UCI SMS Spam Collection without fabricating fallback data."""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from sms_spam_mlops.config import settings
from sms_spam_mlops.data.prepare_data import load_and_validate


def download_dataset(data_dir: Path, url: str, force: bool = False) -> Path:
    """Download and validate the dataset, reusing a valid local copy by default."""

    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / settings.dataset_filename
    if destination.exists() and not force:
        load_and_validate(destination)
        return destination
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Could not download dataset from {url}: {exc}") from exc
    if not payload:
        raise RuntimeError("Dataset download returned an empty response")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next((name for name in archive.namelist() if name.endswith("SMSSpamCollection")), None)
            if member is None:
                raise RuntimeError("Downloaded archive does not contain SMSSpamCollection")
            destination.write_bytes(archive.read(member))
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(f"Downloaded dataset archive is malformed: {exc}") from exc
    load_and_validate(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--url", default=settings.dataset_url)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_dataset(args.data_dir, args.url, args.force))


if __name__ == "__main__":
    main()