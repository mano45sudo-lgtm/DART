import sys


def main() -> int:
    # Ensure repo root is importable when running from scripts/
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import env  # noqa: F401
    import reward  # noqa: F401
    import tools  # noqa: F401

    print("sanity: imports ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

