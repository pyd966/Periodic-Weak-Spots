from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .profiles import QWEN_MODEL_ID, get_model_profile, verify_tokenizer_files


def check_resources(root: str | Path) -> dict:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["model_id"] != QWEN_MODEL_ID:
        raise ValueError("资源清单的版本或模型名称不正确")
    total_bytes = 0
    for relative, entry in manifest["files"].items():
        path = (root / relative).resolve()
        path.relative_to(root)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"文件大小与资源清单不一致：{relative}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"文件 SHA256 与资源清单不一致：{relative}")
        total_bytes += entry["bytes"]
    order = manifest["paul_graham_join_order"]
    if len(order) != manifest["paul_graham_article_count"] or len(set(order)) != len(order):
        raise ValueError("Paul Graham 文章数量或连接顺序不正确")
    reconstructed = "".join((root / relative).read_bytes().decode("utf-8") for relative in order)
    if reconstructed.encode("utf-8") != (root / "niah_128k/paul_graham/filler.txt").read_bytes():
        raise ValueError("文章连接结果与 filler.txt 不一致")
    profile = get_model_profile(QWEN_MODEL_ID)
    verify_tokenizer_files(profile, local_path=root / "qwen3_8_flash_next")
    return {
        "status": "ok",
        "checked_files": len(manifest["files"]),
        "total_bytes": total_bytes,
        "paul_graham_article_count": len(order),
        "paul_graham_source_revision": manifest["paul_graham_source_revision"],
        "tokenizer_revision": profile.tokenizer_revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resources", type=Path, default=Path(__file__).resolve().parents[2] / "resources"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check_resources(args.resources)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
