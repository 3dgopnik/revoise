"""
Download TTS models into the portable cache and place them in
D:\\RevoicePortable\\models\\tts\\...
- Coqui XTTS v2 (multi-language)
"""

import os
from pathlib import Path

HF_REPOS = {
    # XTTS v2 — offline checkpoint (Coqui)
    "coqui_xtts": "coqui/XTTS-v2",  # can be replaced with an alternative fork if desired
}

def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models" / "tts"
    models_dir.mkdir(parents=True, exist_ok=True)

    # HF cache in portable
    hf_home = root / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))

    print("HF cache:", hf_home)

    from huggingface_hub import snapshot_download

    for name, repo in HF_REPOS.items():
        target = models_dir / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}: {repo} -> {target}")
        local = snapshot_download(
            repo_id=repo,
            local_dir=target,
            local_dir_use_symlinks=False,
            revision="main",
            ignore_patterns=["*.md", "*.png", "*.jpg", "*.gif", "*.ipynb"],
        )
        print("  done:", local)

    print("OK. Models downloaded.")

if __name__ == "__main__":
    main()
