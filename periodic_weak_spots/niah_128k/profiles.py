from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-Base"
QWEN_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    tokenizer_id: str
    tokenizer_revision: str
    experiment_id: str
    sample_prefix: str
    model_path: str
    use_chat_template: bool
    tokenizer_files: tuple[tuple[str, str], ...] = ()


MODEL_PROFILES = {
    DEEPSEEK_MODEL_ID: ModelProfile(
        model_id=DEEPSEEK_MODEL_ID,
        tokenizer_id="deepseek-ai/DeepSeek-V4-Flash",
        tokenizer_revision="60d8d70770c6776ff598c94bb586a859a38244f1",
        experiment_id="niah-128k-balanced-target64k-16000pairs",
        sample_prefix="niah128k",
        model_path="/models/DeepSeek-V4-Flash-Base",
        use_chat_template=False,
    ),
    QWEN_MODEL_ID: ModelProfile(
        model_id=QWEN_MODEL_ID,
        tokenizer_id=QWEN_MODEL_ID,
        tokenizer_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
        experiment_id="niah-128k-qwen3.8-flash-next-balanced-target64k-16000pairs",
        sample_prefix="niah128k-qwen38",
        model_path="/models/Qwen3.8-Flash-Next",
        use_chat_template=True,
        tokenizer_files=(
            ("tokenizer.json", "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
            (
                "tokenizer_config.json",
                "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
            ),
            (
                "chat_template.jinja",
                "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
            ),
        ),
    ),
}


def get_model_profile(model_id: str) -> ModelProfile:
    if model_id not in MODEL_PROFILES:
        raise ValueError(f"尚未注册 NIAH 模型：{model_id}")
    return MODEL_PROFILES[model_id]


def verify_tokenizer_files(
    profile: ModelProfile,
    *,
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    hashes: dict[str, str] = {}
    for filename, expected in profile.tokenizer_files:
        if local_path is None:
            path = Path(
                hf_hub_download(
                    profile.tokenizer_id,
                    filename,
                    revision=profile.tokenizer_revision,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            )
        else:
            path = Path(local_path) / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Tokenizer 文件与已注册 revision 不一致：{path}")
        hashes[filename] = actual
    return hashes
