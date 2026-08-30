"""Dataset helper.
BY AYUSH RAJPUT

Recommended academic workflow: configure Kaggle API credentials and run this file,
or download Flickr8k manually and follow README.md.
"""
import subprocess, sys
from pathlib import Path

def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    print("Attempting Kaggle download. If your account does not have access, download Flickr8k manually.")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", "adityajn105/flickr8k", "-p", str(out), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print("Download finished. Inspect data/ and arrange it according to README.md.")
    except Exception as exc:
        print("Automatic download failed:", exc)
        print("Manual dataset placement instructions are in README.md.")

if __name__ == "__main__":
    main()
