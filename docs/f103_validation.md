# F103 구현과 실제 API 검증

2026-10-21. 기존 Python 3.12.10 / Windows 환경에서 F103만 추가했다.
F101·F102 코드는 수정하지 않고 YOLO 검출 결과 변환 함수와 F102 VLM 로더·프롬프트를 재사용한다.
교육 원본은 수정하지 않았으며 F104~F106·광고 정책 RAG는 구현하지 않는다.

## 요구사항과 구현

| 항목 | 구현 |
|---|---|
| JPEG → YOLO 객체 요약 | `POST /api/yolo-detect`, summary/elapsed_ms |
| JPEG → 한국어 VLM 설명 | `POST /api/vlm-describe`, description/elapsed_ms |
| 로딩 상태 확인 | `GET /api/health`, 모델 상태·횟수·실행 설정 |
| 중복 초기화 방지 | 모델별 load lock과 double check, 실패 상태 캐시 |
| 동일 모델 동시 추론 제어 | 모델별 inference lock, YOLO와 VLM 사이에는 별도 락 |
| 이벤트 루프 보호 | JPEG 검증·모델 로딩·추론 모두 `run_in_threadpool` |
| 성능 비교 | 새 서버의 첫 요청, warm 순차·병렬 HTTP 호출을 별도 측정 |

구현 파일은 `app/model_manager.py`, `app/main.py`,
`experiments/f103_api_latency_compare.py`, `tests/test_f103.py`다.
의존성에는 기존 설치된 `httpx==0.28.1`을 명시했다.

서버는 모델을 시작 시 로드하지 않는다. 첫 요청에서 해당 모델만 로드한다.
초기화 실패는 상세 로그와 함께 failed 상태로 보존하므로 수정 후 서버를 재시작해야 한다.
health는 모델 초기화·추론 락을 획득하지 않는다.
VLM은 Q4_K_M 본체와 mmproj를 F102의 `load_vlm()`으로 로드한다.
temperature=0.0, seed=42, 한국어 짧은 프롬프트, max_tokens=320,
n_ctx=8192, n_threads=8, CPU 설정이다. YOLO는 conf=0.25, imgsz=640이다.

## 실행 명령

서버 터미널:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-graceful-shutdown 660
```

다른 터미널에서 방금 시작한 서버를 측정한다. 동일 모델을 이미 요청했다면 먼저 재시작한다.

```powershell
.\.venv\Scripts\python.exe -u -m experiments.f103_api_latency_compare --timeout 600
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실제 측정은 `--output-dir artifacts/f103/baseline`을 지정했다.
기존 결과를 덮어쓰지 않으므로 재실행에는 새 폴더를 사용한다.
원본 `source/2회차_day1/samples/sample.jpg`의 바이트를 매번 동일하게 전송한다.

## 시간의 의미와 타임아웃

- client_ms: HTTP 클라이언트가 POST를 시작해서 응답 본문을 모두 받은 시간.
- api_elapsed_ms: 서버 본문 읽기·JPEG 검증·스레드풀 대기·락 대기·모델 로딩·추론.
  응답 JSON 전송은 포함하지 않는다.
- model_inference_ms: 검증 이후 모델 호출·결과 요약/추출. 모델 로딩·락 대기는 제외한다.
  VLM은 base64 변환·이미지 디코딩/비전 인코딩·입력 평가·텍스트 생성도 포함한다.
- model_load_ms: 해당 요청에서 처음 로드한 시간. 캐시 요청은 0이다.
- model_queue_ms: 모델 추론 락을 기다린 시간. 워커 스레드 배정 대기는 포함하지 않는다.
- total_wall_ms: 순차 또는 병렬 2요청을 완료하는 클라이언트 전체 시간.
  완료 결과의 작은 CSV/진행 출력 기록 비용도 포함한다.

모델별 세부 시간은 응답 헤더에 기록하며 공용 JSON은 지정된 두 키를 유지한다.
client와 server 시간은 측정 구간이 달라 차이를 순수 네트워크 지연으로 단정하지 않는다.

클라이언트의 read/write/pool 타임아웃은 600초, connect는 10초다.
서버의 모델 락 대기 제한은 600초이며 초과하면 503을 반환한다.
이는 전체 실행을 강제 중단하는 제한이 아니다. native 추론이 시작된 뒤 HTTP 연결이 끊겨도
모델 공유 상태를 보호하기 위해 실행 완료까지 락을 유지한다.
Uvicorn graceful shutdown 660초는 종료 시 요청 정리를 기다리는 시간이다.
참고: [Starlette thread pool](https://starlette.dev/threadpool/),
[Uvicorn 옵션](https://www.uvicorn.org/deployment/).

## 단위·동시성 검증

F101 8개 + F102 10개 + F103 10개 = **28개 통과**, 최종 실행 0.802초.
결과는 `artifacts/f103/unit_tests_final.log`에 있다.

F103 검증은 정상 JPEG와 응답 스키마, 빈 입력·손상·비JPEG·잘못된 Content-Type·용량 초과,
초기화 실패 캐시/오류 비노출, 추론 오류 후 락 해제, 큐 타임아웃,
모델별 8스레드 동시 get의 단일 로딩, 동시 VLM HTTP 요청의 단일 로딩/직렬 추론,
YOLO·VLM 동시 진입, VLM 블로킹 중 health 응답, 측정 도구의 순차·병렬 dispatch를 포함한다.
단위 테스트는 가짜 모델·임시 JPEG를 사용했고 테스트 출력 시간을 실제 모델 성능으로 취급하지 않았다.

## 실제 서버의 입력 오류와 이벤트 루프 검증

VLM이 실제 추론 중일 때 GET health는 HTTP 200, client_ms=5.885로 응답했다.
당시 VLM active_inferences=1이었다. 두 API에 빈 입력과 손상 입력을 보낸 4요청은 모두 400이었다.
빈 입력 응답은 `{"error":"JPEG body must not be empty"}`,
손상 입력은 `{"error":"Invalid or corrupted JPEG image"}`였다.
처음 손상 검증은 약 1.027초였고 나머지 오류 응답은 약 1.8~2.8ms였다.
이를 모든 상황의 응답 시간 보장으로 일반화하지 않는다.
실제 결과는 `artifacts/f103/live_checks.json`에 있다.

## 실제 모델 API 측정

첫 요청 1쌍을 워밍업으로 삼고, 이후 순차 1쌍과 병렬 1쌍을 실행했다.
실제 run_id: `20261021T070726Z-3d8b39cd`.
원본·응답·설정·환경 및 시작/종료 health 상태는
`artifacts/f103/baseline/run.json`, `api_requests.csv`, `api_groups.csv`에 보존한다.
측정은 2026-10-21 16:07~16:13(한국 시간)에 수행했다. 결과는 모두 실제 값이며
각 모델의 첫 요청을 워밍업으로 사용했다.

| 시나리오 | YOLO HTTP(ms) | VLM HTTP(ms) | 2요청 전체(s) |
|---|---:|---:|---:|
| cold_sequential | 2,669.864 | 107,478.388 | 110.150 |
| warm_sequential | 54.652 | 112,370.941 | 112.427 |
| warm_parallel | 67.602 | 121,961.247 | 121.964 |

| 시나리오·모델 | 서버 처리(ms) | 모델 추론(ms) | 모델 로딩(ms) |
|---|---:|---:|---:|
| cold YOLO | 2,667.373 | 1,216.546 | 1,428.095 |
| cold VLM | 107,461.193 | 104,956.202 | 2,498.544 |
| warm 순차 YOLO | 51.066 | 44.884 | 0 |
| warm 순차 VLM | 112,366.560 | 112,359.977 | 0 |
| warm 병렬 YOLO | 62.058 | 55.361 | 0 |
| warm 병렬 VLM | 121,953.463 | 121,946.283 | 0 |

6요청 모두 HTTP 200, finish_reason=stop이었다. cold_load는 각 모델의 첫 요청에서만 true였다.
시작 health에서는 두 모델이 모두 unloaded였고, 종료 health에서 두 모델은 각각
load_attempts=1, load_count=1, completed_inferences=3, active_inferences=0이었다.
즉, 첫 요청 이후 캐시 인스턴스를 재사용하면서도 요청마다 실제 추론을 실행했다.

warm 순차 모델 추론 시간의 합에서 VLM 비중은 약 99.96%였다. 병렬 호출에서도 전체 시간은
VLM 응답 시간과 거의 같았다. 이번 병렬 전체 시간은 순차보다 약 9.54초(8.5%) 더 길었으며,
성능 개선을 확인했다고 기록하지 않는다. CPU 공유와 시스템 부하가 있는 1회 측정으로,
차이를 HTTP 병렬화 하나의 인과 효과로 확정할 수 없다. warm 순차 구간에는 위의 실제 health·
잘못된 입력 검증과 가벼운 환경 검사도 병행했다. 부하 통제·반복 실험은 수행하지 않았다.

### 실제 JSON 응답 예시

warm 순차 YOLO 응답의 elapsed_ms를 소수 셋째 자리로 반올림해 표시한다:

```json
{"summary":"bus 1개, person 4개","elapsed_ms":51.066}
```

warm 순차 VLM 응답:

```json
{"description":": 사진에는 흰색 코트를 입은 남성과 검은 코트를 입은 남성이 버스 앞에 서 있는 모습이 보입니다. 버스는 'cero emisiones'라는 문구가 적힌 전기버스로, 배경에는 건물과 나무가 보입니다.","elapsed_ms":112366.560}
```

세 번의 VLM 응답 원문은 일치했다. F102에서 확인한 문장 수·의미 정확도의 제한은 그대로이며
요청한 한국어 한 문장보다 길게 설명했다. 문자열을 보정하거나 샘플별 답을 하드코딩하지 않았다.
위 수치는 예시를 위해 만든 값이 아니라 `api_requests.csv`의 실측 응답에서 가져왔다.
서버와 클라이언트 콘솔 로그는 `artifacts/f103/server_*.log`, `client_*.log`에 남겼다.

## 범위와 제약

모든 측정은 로컬 단일 워커 CPU에서 각 시나리오 1회로 수행한다.
샘플 수가 적어 평균·p95나 통계적 속도 개선을 주장하지 않는다.
워크로드가 큰 VLM과 작은 YOLO가 CPU를 공유하므로 병렬 요청이 더 빠르다고 보장하지 않는다.
HTTP 캐시를 만들지 않으며 매 요청 실제 추론을 실행한다.
모델 초기화 횟수는 프로세스별이다. workers를 늘리면 모델 메모리도 중복된다.
thread pool의 모델별 락 대기는 워커를 사용하므로 요청이 많은 운영 환경의 admission control은 별도 과제다.
한국어 설명의 의미 정확도·문장 수를 보정하지 않으며 생성 문장은 확정된 정답이 아니다.
원본/최적화 설정을 비교하는 F106 실험이나 API 외 광고 심사 기능은 추가하지 않았다.

F101·F102 기존 코드·문서·측정 파일 28개의 SHA256을 작업 전후 비교했다.
기준 목록은 `artifacts/f103/preservation_before.json`, 최종 보존·CSV 검증은
`artifacts/f103/validation_summary.json`에 기록한다.
실제 API·동시성 테스트·측정에서 남은 차단 사항은 없다. F103 실측용 임시 서버는 종료했으며
다시 사용할 때 위 명령으로 실행한다. 네이티브 추론 강제 취소, 큰 요청량의 대기열 제어,
VLM의 장시간 지연과 설명 품질은 현재 PoC의 제한이다.
