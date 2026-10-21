"""Run Chrome browser QA. Mock UI tests and real inference are separate modes."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.config import Settings, project_path


async def check(args):
    from playwright.async_api import async_playwright
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"mode": "real_models" if args.real else "mock_api", "checks": [], "api_responses": []}
    errors = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(executable_path=args.browser, headless=True)
            page = await browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            if args.real:
                # Chrome's inspector may evict response bodies. Observe a native fetch clone
                # in the page instead; do not change the request or substitute its response.
                await page.add_init_script("""(() => {
                  window.__uiActualResponses = [];
                  const originalFetch = window.fetch.bind(window);
                  window.fetch = async (...args) => {
                    const response = await originalFetch(...args);
                    if (args[1]?.method === 'POST' && String(args[0]).includes('/api/')) {
                      const body = await response.clone().json();
                      window.__uiActualResponses.push({url: response.url, status: response.status, body});
                    }
                    return response;
                  };
                })();""")
            if not args.real:
                state = {"mode": "normal", "calls": [], "active": False}

                async def route_api(route):
                    name = "yolo" if "yolo-detect" in route.request.url else "vlm"
                    state["calls"].append(name)
                    assert hashlib.sha256(route.request.post_data_buffer).hexdigest() == hashlib.sha256(args.image.read_bytes()).hexdigest()
                    state["active"] = True
                    await asyncio.sleep(.4 if state["mode"] != "slow" else 2)
                    state["active"] = False
                    if state["mode"] == "error":
                        await route.fulfill(status=400, json={"error": "손상된 JPEG 테스트 오류"})
                    elif name == "yolo":
                        await route.fulfill(json={"summary": "person 1개", "elapsed_ms": 15,
                            "image_width": 810, "image_height": 1080,
                            "detections": [{"bbox": [100, 100, 300, 400], "class_id": 0,
                                            "class": "person", "confidence": .9}]})
                    else:
                        await route.fulfill(json={"description": "버스 옆에 사람이 서 있습니다.", "elapsed_ms": 400})

                await page.route("**/api/yolo-detect*", route_api)
                await page.route("**/api/vlm-describe", route_api)
                await page.route("**/api/health", lambda route: route.fulfill(json={"models": {
                    "vlm": {"status": "loaded", "active_inferences": int(state["active"])}}}))

            await page.goto(args.url)
            await page.locator("h1").wait_for()
            await page.locator("#image-file").set_input_files(str(args.image))
            await page.wait_for_function("!document.querySelector('#preview-frame').hidden && !document.querySelector('#run-both').disabled")
            assert await page.locator("#preview").evaluate("img => img.naturalWidth") == 810
            report["checks"].append("JPEG selection and preview 810x1080")

            if args.real:
                await page.locator("#run-yolo").click()
                await page.wait_for_function("document.querySelector('#yolo-status').textContent === '완료'", timeout=600000)
                assert await page.locator("#object-counts span").count() > 0
                report["checks"].append("Real YOLO summary, counts, elapsed and bbox")
                await page.locator("#run-both").click()
                await page.wait_for_timeout(1200)
                assert await page.locator("#run-both").is_disabled()
                assert "경과" in await page.locator("#vlm-time").inner_text()
                await page.wait_for_function("document.querySelector('#vlm-status').textContent === '완료' && !document.querySelector('#run-both').disabled", timeout=600000)
                assert await page.locator("#vlm-description").inner_text()
                report["checks"].append("Real simultaneous YOLO + VLM, elapsed progress and Korean description")
                await page.wait_for_timeout(200)
                report["api_responses"] = await page.evaluate("window.__uiActualResponses")
                assert len(report["api_responses"]) == 3
                assert all(row["status"] == 200 for row in report["api_responses"])
            else:
                await page.locator("#run-both").click()
                await page.wait_for_timeout(100)
                assert await page.locator("#run-both").is_disabled()
                await page.evaluate("document.querySelector('#run-both').click()")
                await page.wait_for_function("!document.querySelector('#run-both').disabled")
                assert len(state["calls"]) == 2
                assert await page.locator("#yolo-summary").inner_text() == "person 1개"
                assert "버스" in await page.locator("#vlm-description").inner_text()
                assert await page.locator("#bbox-canvas").evaluate("c => c.getContext('2d').getImageData(100,200,1,1).data[3]") > 0
                report["checks"].append("Both APIs, raw bytes, counts, bbox coordinate pixel and duplicate prevention")
                await page.locator("#run-yolo").click()
                await page.wait_for_function("!document.querySelector('#run-yolo').disabled")
                await page.locator("#run-vlm").click()
                await page.wait_for_function("!document.querySelector('#run-vlm').disabled")
                report["checks"].append("YOLO-only and VLM-only actions")
                state["mode"] = "error"
                await page.locator("#run-yolo").click()
                await page.wait_for_function("document.querySelector('#yolo-status').textContent === '오류'")
                assert "손상된 JPEG" in await page.locator("#yolo-summary").inner_text()
                report["checks"].append("API 400 error shown")
                state["mode"] = "slow"
                await page.locator("#run-vlm").click()
                await page.wait_for_timeout(200)
                await page.locator("#cancel").click()
                await page.wait_for_function("document.querySelector('#vlm-status').textContent.includes('취소')")
                assert await page.locator("#run-both").is_disabled()
                await page.wait_for_function("!document.querySelector('#run-both').disabled", timeout=20000)
                report["checks"].append("Browser cancellation, server continuation notice and idle wait")
                await page.locator("#clear").click()
                for name, content, mime in [("empty.jpg", b"", "image/jpeg"), ("bad.png", b"not png", "image/png"),
                                            ("broken.jpg", b"\xff\xd8broken", "image/jpeg")]:
                    await page.locator("#image-file").set_input_files({"name": name, "mimeType": mime, "buffer": content})
                    await page.wait_for_function("document.querySelector('#notice').classList.contains('error')")
                    assert await page.locator("#run-both").is_disabled()
                report["checks"].append("Empty, wrong-format and corrupted image blocked")
                await page.locator("#image-file").set_input_files(str(args.image))
                await page.wait_for_function("!document.querySelector('#preview-frame').hidden")
            await page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
            await page.set_viewport_size({"width": 390, "height": 844})
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
            report["checks"].append("Mobile viewport has no horizontal overflow")
            assert not errors, errors
            report["page_errors"] = errors
            report["passed"] = True
            await browser.close()
    except Exception as error:
        import traceback
        report.update(passed=False, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--image", type=project_path, default=Settings.from_env().image)
    parser.add_argument("--output", type=project_path, required=True)
    parser.add_argument("--browser", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--real", action="store_true")
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
