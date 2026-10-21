"use strict";
const $ = (id) => document.getElementById(id);
const buttons = [$("run-yolo"), $("run-vlm"), $("run-both")];
let file = null, imageUrl = null, busy = false, validating = false, draining = false, generation = 0;
const controllers = new Map(), active = new Map();
const notice = (text, kind = "") => { $("notice").textContent = text; $("notice").className = `notice ${kind}`; };
function controls() {
  buttons.forEach((b) => b.disabled = !file || busy || validating || draining);
  $("image-file").disabled = busy || draining || validating;
  $("clear").disabled = !file || busy || draining || validating;
  $("cancel").hidden = !busy;
}
function reset() {
  file = null;
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null; $("preview").removeAttribute("src");
  $("preview-frame").hidden = true; $("drop-zone").hidden = false;
  $("object-counts").replaceChildren(); $("file-info").textContent = "JPEG · 최대 10 MiB";
  $("dimensions").textContent = "업로드 전 브라우저에서 미리볼 수 있습니다.";
  $("yolo-summary").textContent = "이미지에서 발견한 객체와 개수가 표시됩니다.";
  $("vlm-description").textContent = "사진 속 장면을 한국어로 설명합니다.";
  for (const name of ["yolo", "vlm"]) { $(name + "-status").textContent = "대기"; $(name + "-status").className = ""; $(name + "-time").textContent = "—"; }
  $("vlm-note").textContent = "생성된 설명에는 부정확한 내용이 포함될 수 있습니다.";
}
async function selectFile(candidate) {
  if (busy || draining || validating) return;
  reset(); validating = true; controls();
  notice("이미지를 확인하고 있습니다.");
  try {
    if (!candidate) { notice("JPEG 이미지를 선택해 시작하세요."); return; }
    if (!candidate.size) throw new Error("빈 파일입니다. 다른 JPEG 이미지를 선택하세요.");
    if (candidate.size > 10 * 1024 * 1024) throw new Error("이미지는 최대 10 MiB까지 선택할 수 있습니다.");
    if ((candidate.type && candidate.type !== "image/jpeg") || !/\.jpe?g$/i.test(candidate.name)) throw new Error("JPEG(.jpg, .jpeg) 이미지 파일을 선택하세요.");
    const signature = new Uint8Array(await candidate.slice(0, 2).arrayBuffer());
    if (signature[0] !== 255 || signature[1] !== 216) throw new Error("JPEG 형식이 아니거나 손상된 이미지입니다.");
    imageUrl = URL.createObjectURL(candidate); $("preview").src = imageUrl;
    try { await $("preview").decode(); } catch { throw new Error("이미지를 읽을 수 없습니다. 손상된 파일인지 확인하세요."); }
    const width = $("preview").naturalWidth, height = $("preview").naturalHeight;
    if (width * height > 20000000) throw new Error("이미지는 최대 2천만 픽셀까지 선택할 수 있습니다.");
    $("bbox-canvas").width = width; $("bbox-canvas").height = height;
    $("preview-frame").style.maxWidth = `${460 * width / height}px`;
    $("preview-frame").style.margin = "0 auto";
    $("bbox-canvas").getContext("2d").clearRect(0, 0, width, height);
    file = candidate; $("preview-frame").hidden = false; $("drop-zone").hidden = true;
    $("file-info").textContent = candidate.name; $("dimensions").textContent = `${width} × ${height} px · ${(candidate.size / 1024).toFixed(1)} KiB`;
    notice("미리보기가 준비됐습니다. 실행할 분석을 선택하세요.");
  } catch (error) { reset(); $("image-file").value = ""; notice(error.message, "error"); }
  finally { validating = false; controls(); }
}
function drawBoxes(data) {
  const canvas = $("bbox-canvas"), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = canvas.width / data.image_width, sy = canvas.height / data.image_height;
  const line = Math.max(2, canvas.width / 300), font = Math.max(13, canvas.width / 48);
  ctx.lineWidth = line; ctx.font = `600 ${font}px sans-serif`;
  const counts = new Map();
  for (const item of data.detections || []) {
    if (!Array.isArray(item.bbox) || item.bbox.length !== 4 || !item.bbox.every(Number.isFinite)) continue;
    const [x1,y1,x2,y2] = item.bbox, x = x1*sx, y = y1*sy;
    ctx.strokeStyle = "#18b4c6"; ctx.strokeRect(x,y,(x2-x1)*sx,(y2-y1)*sy);
    const label = `${item.class} ${(item.confidence*100).toFixed(1)}%`, labelWidth = ctx.measureText(label).width + 12;
    const labelX = Math.max(0, Math.min(x, canvas.width - labelWidth)), labelY = Math.max(font + 10, y);
    ctx.fillStyle = "#073e55"; ctx.fillRect(labelX, labelY-font-9, labelWidth, font+9);
    ctx.fillStyle = "#fff"; ctx.fillText(label, labelX+6, labelY-5);
    counts.set(item.class, (counts.get(item.class)||0)+1);
  }
  $("object-counts").replaceChildren();
  for (const [name,count] of counts) { const badge = document.createElement("span"); badge.textContent = `${name} · ${count}개`; $("object-counts").append(badge); }
}
async function analyze(name, token) {
  const controller = new AbortController(); controllers.set(name, controller);
  active.set(name, performance.now()); $(name+"-status").textContent = "분석 중"; $(name+"-status").className = "running";
  const timeout = setTimeout(() => { controller.uiTimedOut = true; controller.abort(); }, 600000);
  try {
    const endpoint = name === "yolo" ? "/api/yolo-detect?include_detections=true" : "/api/vlm-describe";
    const response = await fetch(endpoint, {method:"POST", headers:{"Content-Type":"image/jpeg"}, body:file, signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `요청에 실패했습니다 (HTTP ${response.status}).`);
    if (token !== generation) return;
    if (!Number.isFinite(data.elapsed_ms) || typeof data[name === "yolo" ? "summary" : "description"] !== "string") throw new Error("API 응답 형식을 확인할 수 없습니다.");
    $(name+"-status").textContent = "완료"; $(name+"-status").className = "done";
    $(name+"-time").textContent = `서버 ${(data.elapsed_ms/1000).toFixed(2)}초`;
    if (name === "yolo") { $("yolo-summary").textContent = data.summary; drawBoxes(data); }
    else { $("vlm-description").textContent = data.description; if (response.headers.get("X-Finish-Reason") === "length") $("vlm-note").textContent = "출력 토큰 한도에 도달해 설명이 잘렸을 수 있습니다."; }
    return "ok";
  } catch (error) {
    $(name+"-status").className = "error";
    if (error.name === "AbortError") { draining = true; $(name+"-status").textContent = controller.uiTimedOut ? "대기 시간 초과" : "응답 대기 취소"; $(name+"-time").textContent = "서버 상태 확인 중"; return "cancelled"; }
    $(name+"-status").textContent = "오류"; $(name+"-time").textContent = "—";
    $(name === "yolo" ? "yolo-summary" : "vlm-description").textContent = error.message;
    return "error";
  } finally { clearTimeout(timeout); active.delete(name); controllers.delete(name); }
}
async function waitForIdle() {
  notice("브라우저 응답 대기를 취소했습니다. 서버 추론은 계속될 수 있어, 서버가 유휴 상태인지 확인할 때까지 새 요청을 잠급니다.", "warning");
  let idle = 0;
  while (draining) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    try {
      const response = await fetch("/api/health", {signal:AbortSignal.timeout(5000), cache:"no-store"});
      if (!response.ok) throw new Error("health failed");
      const health = await response.json();
      const working = Object.values(health.models).some(model => model.status === "loading" || model.active_inferences > 0);
      idle = working ? 0 : idle+1;
      if (idle >= 2) { draining = false; notice("서버가 유휴 상태입니다. 새 분석을 실행할 수 있습니다. 취소한 요청의 결과는 표시하지 않습니다.", "warning"); }
    } catch { idle = 0; notice("서버 상태를 확인할 수 없어 새 요청이 잠겨 있습니다. 서버 연결을 확인하세요. 브라우저 취소는 서버 추론 중단을 보장하지 않습니다.", "warning"); }
  }
  controls();
}
async function run(names) {
  if (!file || busy || validating || draining) return;
  busy = true; controls(); const token = ++generation;
  notice("분석 중입니다. VLM은 CPU에서 수 분이 걸릴 수 있습니다.");
  for (const name of names) {
    if (name === "yolo") { $("bbox-canvas").getContext("2d").clearRect(0,0,$("bbox-canvas").width,$("bbox-canvas").height); $("object-counts").replaceChildren(); $("yolo-summary").textContent = "객체를 분석하고 있습니다."; }
    else { $("vlm-description").textContent = "장면 설명을 생성하고 있습니다."; $("vlm-note").textContent = "생성된 설명에는 부정확한 내용이 포함될 수 있습니다."; }
  }
  const results = await Promise.all(names.map(name => analyze(name, token)));
  busy = false; controls();
  if (draining) await waitForIdle();
  else notice(results.every(result => result === "ok") ? "분석이 완료됐습니다." : "일부 분석에 실패했습니다. 각 결과의 오류 내용을 확인하세요.", results.includes("error") ? "error" : "");
}
setInterval(() => { for (const [name,start] of active) $(name+"-time").textContent = `경과 ${((performance.now()-start)/1000).toFixed(1)}초`; }, 100);
$("image-file").addEventListener("change", event => selectFile(event.target.files[0]));
$("clear").addEventListener("click", () => { reset(); $("image-file").value = ""; controls(); notice("JPEG 이미지를 선택해 시작하세요."); });
$("drop-zone").addEventListener("dragover", event => event.preventDefault());
$("drop-zone").addEventListener("drop", event => { event.preventDefault(); selectFile(event.dataTransfer.files[0]); });
$("run-yolo").addEventListener("click", () => run(["yolo"]));
$("run-vlm").addEventListener("click", () => run(["vlm"]));
$("run-both").addEventListener("click", () => run(["yolo","vlm"]));
$("cancel").addEventListener("click", () => { for (const controller of controllers.values()) controller.abort(); });
window.addEventListener("beforeunload", () => { if (imageUrl) URL.revokeObjectURL(imageUrl); });
