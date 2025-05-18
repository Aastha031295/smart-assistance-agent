import subprocess
import sys
from pathlib import Path


def main():
    print("🔍 Running linting tools...")

    # Get the project root directory
    root_dir = Path(__file__).parent.parent.parent

    # Run isort first
    print("\n🔄 Running isort to sort imports...")
    isort_result = subprocess.run(
        ["isort", str(root_dir)], capture_output=True, text=True
    )

    if isort_result.returncode != 0:
        print(f"❌ isort failed with error:\n{isort_result.stderr}")
        sys.exit(isort_result.returncode)

    if isort_result.stdout:
        print(f"✅ isort output:\n{isort_result.stdout}")
    else:
        print("✅ isort completed successfully!")

    # Then run black
    print("\n🔤 Running black to format code...")
    black_result = subprocess.run(
        ["black", str(root_dir)], capture_output=True, text=True
    )

    if black_result.returncode != 0:
        print(f"❌ black failed with error:\n{black_result.stderr}")
        sys.exit(black_result.returncode)

    if black_result.stdout:
        print(f"✅ black output:\n{black_result.stdout}")
    else:
        print("✅ black completed successfully!")

    print("\n✨ All linting completed successfully! ✨")
    return 0


if __name__ == "__main__":
    sys.exit(main())
