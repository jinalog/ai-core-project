import base64
import csv
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.f102_vlm_prompt_compare import (
    PROMPTS, image_data_url, load_vlm, main, messages_for, reproducibility,
    run_comparison, text_metrics, validate_inputs, write_csv,
)


def answer(text="버스 옆에 사람들이 서 있습니다.", finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 20}}


class F102Tests(unittest.TestCase):
    def test_messages_contain_original_image_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.jpg"
            image.write_bytes(b"test-only-image-bytes")
            url = image_data_url(image)
            self.assertEqual(base64.b64decode(url.split(",")[1]), image.read_bytes())
            messages = messages_for(PROMPTS[0], url)
            self.assertEqual(messages[1]["content"][0]["image_url"]["url"], url)
            self.assertEqual(messages[1]["content"][1]["text"], PROMPTS[0].text)
            self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_repeats_parameters_timing_and_checkpoint(self):
        llm = Mock()
        llm.create_chat_completion.side_effect = [answer()] * 4 + [answer("People stand beside a bus.")] * 2
        ticks = iter([0, 2, 10, 13, 20, 24, 30, 35, 40, 46, 50, 57])
        checkpoint = Mock()
        rows = run_comparison(llm, "data:test", clock=lambda: next(ticks), on_row=checkpoint)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["repeat"] for r in rows], [1, 2] * 3)
        self.assertEqual([r["elapsed_ms"] for r in rows], [2000, 3000, 4000, 5000, 6000, 7000])
        self.assertEqual(checkpoint.call_count, 6)
        calls = llm.create_chat_completion.call_args_list
        for index in range(0, 6, 2):
            self.assertEqual(calls[index], calls[index + 1])
        for call in calls:
            self.assertEqual(call.kwargs["temperature"], 0.0)
            self.assertEqual(call.kwargs["seed"], 42)
            self.assertEqual(call.kwargs["max_tokens"], 320)
            self.assertFalse(call.kwargs["stream"])
        self.assertTrue(all(r["language_matches_heuristic"] for r in rows))
        self.assertTrue(all(r["reproducible_complete"] for r in reproducibility(rows)))

    def test_reproducibility_handles_differences_truncation_and_singletons(self):
        rows = [{"prompt_id": "different", "response": text, "finish_reason": "stop"}
                for text in ("버스", "사람")]
        rows += [{"prompt_id": "truncated", "response": "버스", "finish_reason": "length"}] * 2
        rows += [{"prompt_id": "single", "response": "버스", "finish_reason": "stop"}]
        results = reproducibility(rows)
        self.assertFalse(results[0]["exact_match"])
        self.assertTrue(results[1]["exact_match"])
        self.assertFalse(results[1]["all_completed"])
        self.assertTrue(all(not r["reproducible_complete"] for r in results))

    def test_language_length_and_empty_text(self):
        self.assertEqual(text_metrics("버스 옆 사람", "ko")["language_hint"], "ko")
        self.assertEqual(text_metrics("A yellow bus.", "en")["word_count"], 3)
        self.assertEqual(text_metrics("123", "ko")["language_hint"], "undetermined")
        llm = Mock()
        for text in ("", "   ", None):
            llm.create_chat_completion.return_value = answer(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                run_comparison(llm, "data:test")

    def test_invalid_settings_and_inference_failure(self):
        for options in ({"repeats": 1}, {"max_tokens": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_comparison(Mock(), "data:test", **options)
        llm = Mock()
        llm.create_chat_completion.side_effect = RuntimeError("test inference failure")
        with self.assertRaisesRegex(RuntimeError, "test inference failure"):
            run_comparison(llm, "data:test")

    def test_input_validation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, projector, image = root / "model.gguf", root / "mmproj.gguf", root / "image.jpg"
            with self.assertRaises(FileNotFoundError):
                validate_inputs(model, projector, image)
            model.write_bytes(b"GGUFtest-only")
            projector.write_bytes(b"GGUFtest-only")
            Image.new("RGB", (2, 2)).save(image, format="JPEG")
            validate_inputs(model, projector, image)
            projector.write_bytes(b"bad-header")
            with self.assertRaisesRegex(ValueError, "Not a GGUF"):
                validate_inputs(model, projector, image)
            projector.write_bytes(b"GGUFtest-only")
            image.write_bytes(b"broken JPEG")
            with self.assertRaises(OSError):
                validate_inputs(model, projector, image)

    def test_projector_load_is_measured_explicitly(self):
        handler, llm = Mock(), Mock()
        with patch("llama_cpp.__version__", "0.3.34"), \
             patch("llama_cpp.Llama", return_value=llm) as factory, \
             patch("llama_cpp.llama_chat_format.Qwen25VLChatHandler", return_value=handler), \
             patch("experiments.f102_vlm_prompt_compare.time.perf_counter", side_effect=[0, 1, 2, 5, 6, 8]):
            result, returned_handler, timing = load_vlm(Path("model.gguf"), Path("mmproj.gguf"),
                                                        n_ctx=8192, n_threads=8, seed=42)
        self.assertIs(result, llm)
        self.assertIs(returned_handler, handler)
        handler._init_mtmd_context.assert_called_once_with(llm)
        self.assertEqual(timing["model_load_ms"], 3000)
        self.assertEqual(timing["mmproj_load_ms"], 2000)
        self.assertEqual(timing["total_load_ms"], 6000)
        self.assertEqual(factory.call_args.kwargs["n_gpu_layers"], 0)

    def test_loader_failure_releases_resources(self):
        handler, llm = Mock(), Mock()
        handler._init_mtmd_context.side_effect = RuntimeError("bad projector")
        with patch("llama_cpp.Llama", return_value=llm), \
             patch("llama_cpp.llama_chat_format.Qwen25VLChatHandler", return_value=handler):
            with self.assertRaisesRegex(RuntimeError, "bad projector"):
                load_vlm(Path("model.gguf"), Path("mmproj.gguf"), n_ctx=8192, n_threads=8, seed=42)
        handler._exit_stack.close.assert_called_once()
        llm.close.assert_called_once()

    def test_csv_preserves_korean_newlines_and_quotes(self):
        text = '버스가 보입니다.\n사람이 "옆"에 있습니다.'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            write_csv(path, [{"response": text, "elapsed_ms": 123}])
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["response"], text)

    def test_cli_failure_records_actual_error_without_fake_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "failed"
            try:
                with patch("sys.argv", ["f102", "--output-dir", str(out), "--model", str(out / "missing.gguf")]):
                    self.assertEqual(main(), 1)
                report = json.loads((out / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error_type"], "FileNotFoundError")
                self.assertEqual(report["completed_inferences"], 0)
                self.assertFalse((out / "prompt_comparison.csv").exists())
                self.assertIn("Traceback", (out / "execution.log").read_text(encoding="utf-8"))
            finally:
                for handler in logging.getLogger().handlers[:]:
                    handler.close()
                    logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
