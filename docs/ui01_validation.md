# UI-01 이미지 업로드 웹 UI

기존 F101~F104를 수정하지 않고 `ui01/` 디렉터리에 UI와 FastAPI 확장 진입점을 추가했다.
HTML/CSS/JavaScript만 사용하며 프런트엔드 빌드나 외부 웹 호스팅은 필요하지 않다.
광고 심사·RAG·LLM Agent는 구현하지 않는다.

## 실행

프로젝트 루트 `C:\work\core_pjt`에서 기존 Python 3.12 가상환경으로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn ui01.server:app --host 127.0.0.1 --port 8001 --workers 1 --timeout-graceful-shutdown 660
```

브라우저에서 `http://127.0.0.1:8001/`을 연다. `GET /`은 index.html을 반환한다.
`/docs` 및 기존 두 POST API·health도 같은 서버에서 제공한다.
기존 `app.main:app` 진입점은 그대로 보존했으므로 그 명령을 사용하면 기존 API 서버가 실행된다.
UI는 `ui01.server:app` 진입점을 사용해야 한다. 기존 8000번 서버를 방해하지 않도록 실측에는
8001을 사용했으며 다른 빈 포트도 지정할 수 있다. 같은 모델의 메모리를 중복 사용하지 않으려면
일반 사용 시 필요한 서버 인스턴스 하나만 실행한다. 각 실행은 workers=1이다.

## 화면 기능

- JPEG 선택 또는 드래그 앤 드롭 → 브라우저 로컬 미리보기. 선택만으로 추론 요청을 보내지 않는다.
- YOLO 분석, VLM 설명, 함께 분석 버튼. 함께 분석은 기존 두 API를 병렬 호출한다.
- YOLO summary, 객체별 개수, 서버 elapsed_ms, 원본 픽셀 기준 bbox와 confidence를 표시한다.
- VLM 한국어 description과 서버 elapsed_ms를 표시한다. 진행 중에는 실제 브라우저 경과 시간을 갱신한다.
- 세로 이미지 미리보기는 높이를 제한하고 종횡비를 유지한다. 캔버스도 같은 비율로 표시한다.
- 빈 파일·비JPEG·손상 파일·10 MiB/2천만 픽셀 제한을 브라우저에서 검사하고 서버 오류도 표시한다.
- 실행 중에는 파일 선택·지우기·분석 버튼을 잠그고 함수 진입점에서도 중복 실행을 방지한다.
- 취소 버튼은 AbortController로 브라우저 응답 대기를 취소한다. 서버 추론을 중단했다고 표시하지 않는다.
  이후 health를 조회해 로딩/추론 중인 모델이 없는 상태를 연속 2회 확인할 때까지 새 분석을 잠근다.
- 브라우저 대기 타임아웃은 600초다. 타임아웃에도 같은 서버 상태 확인 절차를 적용한다.
- 설명과 오류 문자열은 textContent, bbox 라벨은 canvas 텍스트로 표시한다.
  선택 이미지와 추론 결과를 이 UI가 파일로 저장하지 않는다.

## API 호환과 모델 관리

기존 `app/main.py`, `app/model_manager.py`를 수정하지 않았다. 확장 앱은 기존 create_app()을
사용하고 같은 URL의 YOLO 라우트에 선택적 쿼리 옵션만 연결한다.

```text
POST /api/yolo-detect
→ 기존 summary, elapsed_ms 두 필드 유지

POST /api/yolo-detect?include_detections=true
→ summary, elapsed_ms, detections, image_width, image_height

POST /api/vlm-describe
→ 기존 description, elapsed_ms 유지
```

확장 응답의 detections 각 항목은 `bbox: [x1,y1,x2,y2]`, `class_id`, `class`, `confidence`다.
좌표는 image_width/image_height 기준 원본 픽셀이다. Pydantic 스키마에 선택 필드를 정의했다.
검출 데이터는 같은 실제 YOLO predict 결과에서 얻으며 bbox를 다시 추정하거나 별도 모델을 로드하지 않는다.
요청별 스레드 로컬 저장소로 캡처 데이터를 분리한다. 실제 모델 캐시·로딩 락·추론 락·
run_in_threadpool 동작은 기존 ModelManager를 그대로 재사용한다.
VLM API는 원래 라우트를 그대로 사용한다. F104 best.pt로 기본 모델을 바꾸지 않았다.

## 자동 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check ui01/static/app.js
```

기존 37개 + UI 확장 API 4개, **총 41개 통과**. 최종 단위 테스트는 1.255초였다.
GET /·정적 자산, 모델 미로딩 상태, 기본 API 두 필드 유지, 선택적 bbox·원본 크기,
VLM 응답 호환성, 잘못된 입력 400과 미로딩을 검사했다.
테스트 모델·이미지는 가짜 객체와 임시 JPEG이며 실제 모델 결과와 구분했다.
종료 시 llama-cpp 계열 import에서 `nul` 파일 ResourceWarning이 관찰됐지만 테스트 실패는 없었다.
최종 JavaScript 문법 검사도 통과했다.

### 실제 Chrome을 사용한 UI 동작 검증

브라우저 자동화는 선택 테스트 의존성이다. 이미 설치된 Chrome을 사용했으며 브라우저를 추가 다운로드하지 않았다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r ui01/requirements-test.txt
# 위의 UI 서버를 먼저 실행한 상태에서 수행
.\.venv\Scripts\python.exe -m ui01.browser_check --output artifacts/ui01/my_mock_check
.\.venv\Scripts\python.exe -m ui01.browser_check --real --output artifacts/ui01/my_real_check
```

Playwright 1.64.0, 로컬 Chrome, 1440×1000 및 390×844 뷰포트를 사용했다.
다른 브라우저 경로는 `--browser`로 지정한다. 출력 디렉터리는 새 경로만 허용한다.
원본 이미지는 기존 `source/2회차_day1/samples/sample.jpg`를 읽었고 교육 파일을 수정하지 않았다.

API를 테스트 응답으로 대체한 브라우저 검증은 `browser_mock_final/report.json`에 **passed=true**다.
검증 항목:

1. 파일 선택 후 810×1080 미리보기 표시 및 원본 JPEG 바이트 전송.
2. YOLO·VLM 개별 버튼과 함께 분석, summary/description/시간·객체 개수 표시.
3. bbox 좌표 (100,200)의 캔버스 알파 픽셀로 실제 선 그리기 확인.
4. 분석 중 버튼 비활성화와 강제 중복 클릭에도 API 호출 수 증가 없음.
5. 서버 400 오류 메시지 표시.
6. 브라우저 취소·서버 계속 가능 안내·유휴 대기 후 버튼 복원.
7. 빈 파일·비JPEG·손상 JPEG 거부, 모바일 가로 넘침 없음, 콘솔 오류 없음.

이 모드의 응답·시간·bbox는 **테스트용 값**이며 실제 모델 성능으로 보고하지 않는다.

### 실제 모델을 사용하는 브라우저 검증

최종 기록 `artifacts/ui01/browser_real_verified/report.json`은 **passed=true**다.
실제 화면에서 YOLO 단독 1회와 함께 분석 1회(YOLO+VLM)를 실행했다.

| 요청 | 실제 서버 elapsed_ms | 결과 |
|---|---:|---|
| YOLO 단독 | 52.341 | bus 1개, person 4개, bbox 5개 |
| 함께 분석 중 YOLO | 83.254 | bus 1개, person 4개, bbox 5개 |
| 함께 분석 중 VLM | 117,168.604 | 한국어 장면 설명 |

3응답 모두 HTTP 200이었다. 버스 confidence 약 0.9402와 인물 4명의 confidence·bbox가 표시됐고,
VLM 진행 중 버튼 잠금과 경과 시간, 완료 후 한국어 설명 및 서버 117.17초 표시를 확인했다.
예측은 원본 모델의 실제 결과이며 하드코딩·API 대체를 하지 않았다.
1440×1000·390×844 캡처를 저장했고 가로 넘침·콘솔 오류가 없었다.

실제 VLM 응답 원문:

> : 사진에는 흰색 코트를 입은 남성과 검은 코트를 입은 남성이 버스 앞에 서 있는 모습이 보입니다. 버스는 'cero emisiones'라는 문구가 적힌 전기버스로, 배경에는 건물과 나무가 보입니다.

UI 변경 후 첫 브라우저 검증은 화면 표시까지 진행됐지만 네트워크 응답 기록이 비어 실패했다.
다음 시도에서는 Chrome 검사기가 응답 본문을 캐시에서 제거했다는 실제 Protocol error를 확인했다.
기록을 삭제하지 않고 `browser_real/`, `browser_real_final/`에 보존했다.
검증 도구를 페이지 안의 실제 fetch 응답 clone을 읽는 방식으로 수정하고 새 출력 폴더에서 재검증했다.
최종 기록은 원문 응답을 대체하지 않으며, 앞선 실패를 성공으로 집계하지 않는다.

## 보존·범위·남은 제한

기존 app/·F101~F104 코드와 테스트·문서·이미지·측정 CSV/JSON의 SHA256 기준 53개를 비교한다.
기준은 `artifacts/ui01/preservation_before.json`, 최종 결과는 `validation_summary.json`에 기록했다.
기존 README·pyproject와 교육 원본은 수정하지 않았다. 신규 UI 파일과 본 문서만 추가한다.
실제/모의 브라우저 캡처는 교육 샘플을 포함하므로 기존 artifacts Git 제외 규칙 안에만 보관한다.

UI-01의 요청된 기능에서 남은 실행 차단 사항은 없다. 다만 다음은 검증 범위의 제한이다.

- 실제 CPU VLM은 약 117초가 걸렸으며 웹 UI 추가로 추론 속도가 개선된 것은 아니다.
- 취소 동작은 Chrome의 테스트 응답으로 검증했다. 장시간 실제 VLM을 중간 취소하는 추가 실험은 하지 않았다.
- 중복 요청 방지는 현재 페이지 기준이다. 다른 탭·클라이언트의 요청은 기존 서버 모델 락으로 제어한다.
- EXIF 회전 사진의 bbox 정합과 대용량·다중 사용자 부하 테스트는 수행하지 않았다.
- 생성 설명의 문장 수·의미 정확도는 기존 F102의 한계를 유지한다.
- 신규 UI는 저장소 루트에서 실행하는 소스 구성이며 wheel 배포 설정은 변경하지 않았다.

검증용 8001 서버는 테스트 후 종료했다. 기존 8000 API 서버에는 변경을 가하지 않았다.
