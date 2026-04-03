from __future__ import annotations

from .config import Settings
from .downloader import ensure_all_downloads


def main() -> None:
    settings = Settings.from_env()
    ensure_all_downloads(settings)
    enabled = ", ".join(spec.alias for spec in settings.enabled_models)
    print(f"Downloads ready for: {enabled}")
    if settings.enable_aligner:
        print(f"Aligner ready at: {settings.aligner_path}")


if __name__ == "__main__":
    main()
