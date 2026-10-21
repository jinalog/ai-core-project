"""F102: local Qwen2.5-VL GGUF prompt comparison, with measured timings."""

import argparse
import base64
import csv
import hashlib
import json
import logging
import platform
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, project_path

SYSTEM_PROMPT = "Describe only visible image content. Follow the requested language and format."


@dataclass(frozen=True)
class Prompt:
    id: str
    language: str
    text: str


PROMPTS = (
    Prompt("ko_short", "ko", "이 사진에서 보이는 주요 장면을 한국어 한 문장으로 간결하게 설명하세요."),
    Prompt("ko_detail", "ko", "이 사진에서 보이는 주요 물체, 사람들의 행동, 공간적 배치를 한국어 세 문장으로 구체적으로 설명하세요. 보이지 않는 사실은 추측하지 마세요."),
    Prompt("en_short", "en", "Describe the main visible scene in one concise English sentence."),
)


def validate_inputs(model: Path, mmproj: Path, image: Path) -> None:
    for path in (model, mmproj, image):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty local file: {path}")
    for path in (model, mmproj):
        with path.open("rb") as stream:
            if path.suffix.lower() != ".gguf" or stream.read(4) != b"GGUF":
                raise ValueError(f"Not a GGUF file: {path}")
    from PIL import Image
    with Image.open(image) as opened:
        if opened.format != "JPEG":
            raise ValueError("Expected an actual JPEG sample")
        opened.verify()


def image_data_url(image: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")


def messages_for(prompt: Prompt, data_url: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt.text}]}]


def text_metrics(text: str, expected_language: str) -> dict:
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    hint = ("ko" if hangul > latin else "en" if latin > hangul else "undetermined")
    return {"char_count": len(text), "word_count": len(text.split()),
            "hangul_chars": hangul, "latin_chars": latin, "language_hint": hint,
            "language_matches_heuristic": hint == expected_language}


def run_comparison(llm, data_url: str, *, prompts=PROMPTS, repeats=2, max_tokens=320,
                   seed=42, clock=time.perf_counter, on_row=lambda row: None) -> list[dict]:
    if repeats < 2 or max_tokens < 1:
        raise ValueError("F102 requires at least two repeats and positive max_tokens")
    rows = []
    for prompt in prompts:
        for repeat in range(1, repeats + 1):
            logging.info("Inference starting: %s repeat %s/%s", prompt.id, repeat, repeats)
            start = clock()
            response = llm.create_chat_completion(messages=messages_for(prompt, data_url),
                                                  temperature=0.0, seed=seed, max_tokens=max_tokens,
                                                  stream=False)
            elapsed_ms = (clock() - start) * 1000
            choice = response["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-text VLM response: {prompt.id}/{repeat}")
            usage = response.get("usage") or {}
            row = {"prompt_id": prompt.id, "expected_language": prompt.language,
                   "prompt": prompt.text, "repeat": repeat, "temperature": 0.0,
                   "seed": seed, "max_tokens": max_tokens, "elapsed_ms": elapsed_ms,
                   "response": text, "finish_reason": choice.get("finish_reason", "unknown"),
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens"),
                   **text_metrics(text, prompt.language)}
            rows.append(row)
            on_row(row)
            logging.info("Inference complete: %s/%s %.3f ms", prompt.id, repeat, elapsed_ms)
    return rows


def reproducibility(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["prompt_id"], []).append(row)
    comparisons = []
    for prompt_id, group in groups.items():
        exact = len({row["response"] for row in group}) == 1
        complete = all(row["finish_reason"] == "stop" for row in group)
        comparisons.append({"prompt_id": prompt_id, "runs": len(group),
                            "exact_match": exact, "all_completed": complete,
                            "reproducible_complete": len(group) >= 2 and exact and complete})
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No inference rows to write")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_vlm(model: Path, mmproj: Path, *, n_ctx: int, n_threads: int, seed: int):
    import llama_cpp
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    if llama_cpp.__version__ != "0.3.34":
        raise RuntimeError(f"Expected llama-cpp-python 0.3.34, got {llama_cpp.__version__}")
    start = time.perf_counter()
    handler = Qwen25VLChatHandler(clip_model_path=str(mmproj), verbose=False)
    handler_setup_ms = (time.perf_counter() - start) * 1000
    llm = None
    try:
        start = time.perf_counter()
        llm = llama_cpp.Llama(model_path=str(model), chat_handler=handler, n_ctx=n_ctx,
                             n_threads=n_threads, n_threads_batch=n_threads, seed=seed,
                             n_gpu_layers=0, logits_all=False, verbose=False)
        model_load_ms = (time.perf_counter() - start) * 1000
        # In 0.3.34 the projector is lazy. Initialize it here so load timing is real.
        start = time.perf_counter()
        handler._init_mtmd_context(llm)
        mmproj_load_ms = (time.perf_counter() - start) * 1000
    except Exception:
        handler._exit_stack.close()
        if llm is not None:
            llm.close()
        raise
    return llm, handler, {"handler_setup_ms": handler_setup_ms,
                           "mmproj_load_ms": mmproj_load_ms, "model_load_ms": model_load_ms,
                           "total_load_ms": handler_setup_ms + mmproj_load_ms + model_load_ms}


def main() -> int:
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=project_path, default=config.vlm_model)
    parser.add_argument("--mmproj", type=project_path, default=config.mmproj)
    parser.add_argument("--image", type=project_path, default=config.image)
    parser.add_argument("--output-dir", type=project_path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--prompt-id", action="append", choices=[p.id for p in PROMPTS],
                        help="Run only selected prompt(s); default: all three")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected_prompts = tuple(p for p in PROMPTS if not args.prompt_id or p.id in args.prompt_id)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.output_dir or project_path(f"artifacts/f102/{run_id}")
    # A new directory prevents failed/partial runs from overwriting older evidence.
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "execution.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    report = {"run_id": run_id, "status": "validating", "python": sys.version,
              "os": platform.platform(), "device": "cpu", "llama_cpp_version": None,
              "model": str(args.model), "mmproj": str(args.mmproj), "image": str(args.image),
              "n_ctx": args.n_ctx, "n_threads": args.n_threads, "seed": args.seed,
              "max_tokens": args.max_tokens, "repeats": args.repeats, "temperature": 0.0,
              "system_prompt": SYSTEM_PROMPT, "prompts": [asdict(p) for p in selected_prompts],
              "completed_inferences": 0}
    llm, handler = None, None
    try:
        save_json(out / "run.json", report)
        if args.repeats < 2 or min(args.max_tokens, args.n_ctx, args.n_threads) < 1:
            raise ValueError("repeats must be >= 2; tokens, context and threads must be positive")
        validate_inputs(args.model, args.mmproj, args.image)
        import llama_cpp
        from PIL import Image
        report["llama_cpp_version"] = llama_cpp.__version__
        report["gpu_offload_supported"] = bool(llama_cpp.llama_supports_gpu_offload())
        report["backend_info"] = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
        report["files"] = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                           for name, path in (("model", args.model), ("mmproj", args.mmproj),
                                              ("image", args.image))}
        with Image.open(args.image) as image:
            report["image_size"] = list(image.size)
        data_url = image_data_url(args.image) # Original JPEG; no resize or alternate input.
        report["status"] = "loading"
        save_json(out / "run.json", report)
        logging.info("Loading local mmproj and model; output directory: %s", out)
        load_start = time.perf_counter()
        try:
            llm, handler, timings = load_vlm(args.model, args.mmproj, n_ctx=args.n_ctx,
                                             n_threads=args.n_threads, seed=args.seed)
        finally:
            report["load_attempt_ms"] = (time.perf_counter() - load_start) * 1000
        report.update(timings)
        report["status"] = "inferencing"
        save_json(out / "run.json", report)
        logging.info("Model loaded: %s", timings)
        saved_rows = []

        def checkpoint(row):
            saved_rows.append({"run_id": run_id, **timings, **row})
            write_csv(out / "prompt_comparison.csv", saved_rows)
            report["completed_inferences"] = len(saved_rows)
            save_json(out / "run.json", report)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        rows = run_comparison(llm, data_url, prompts=selected_prompts,
                              repeats=args.repeats, max_tokens=args.max_tokens,
                              seed=args.seed, on_row=checkpoint)
        comparisons = reproducibility(rows)
        write_csv(out / "reproducibility.csv", comparisons)
        report.update(status="completed", reproducibility=comparisons)
        save_json(out / "run.json", report)
        logging.info("F102 completed: %s", out)
        return 0
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save_json(out / "run.json", report)
        logging.exception("F102 failed; completed results, if any, are retained: %s", out)
        return 1
    finally:
        if handler is not None:
            handler._exit_stack.close()
        if llm is not None:
            llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
