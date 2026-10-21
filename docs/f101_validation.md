# 환경 및 F101 검증 결과

2026-10-21(한국 시간). 로컬 명세서의 F101 필수 조건을 확인한 뒤 검증했다.

## 환경

- Windows 11, Python 3.12.10, Torch 2.14.0+cpu. CUDA 사용 불가.
- Ultralytics 8.4.164, llama_cpp 0.3.34, FastAPI 0.141.1, Uvicorn 0.54.0,
  huggingface_hub 2.0.0, psutil 7.2.2, Pillow 12.3.0 import 성공.
- 기존 가상환경 재사용, 필수 모델 파일·샘플 존재 확인 성공.
- 로컬 editable 설치 성공, `pip check`: `No broken requirements found.`
- `artifacts/environment.json`에 환경과 RAM 기록.

## 단위 테스트

`python -m unittest discover -s tests -v`: **8개 통과**, 0.124초.
검증 항목은 bbox/class/confidence 변환, 무검출, 통계·FPS·p95, 워밍업 제외,
세 conf 전달, 동기화 호출, RSS 증가량, CSV 왕복 읽기, 잘못된 인자·누락 입력·손상 JPEG·빈 CSV다.
테스트의 가짜 모델·시계는 임시 검증에만 사용했다.

## 실제 F101 실행

기존 `yolo11n.pt`와 제공 샘플의 원래 경로를 사용했다. 이미지 크기는 810×1080,
imgsz=640, device=cpu, conf당 warmup=1 및 측정=10회다.
run_id는 `2026-10-21T06:18:24.259369+00:00`이다.

| conf | 검출 수 | 평균 지연(ms) | FPS | RSS 증가(MiB) |
|---:|---:|---:|---:|---:|
| 0.10 | 6 | 39.792 | 25.131 | 138.883 |
| 0.25 | 5 | 37.270 | 26.832 | 0.219 |
| 0.60 | 5 | 38.385 | 26.052 | 3.855 |

RSS 증가는 각 conf 시작 전 대비 반복 종료 후의 프로세스 RSS 차이다.
첫 conf의 초기 할당을 포함하며 피크 메모리가 아니다. 모든 수치는 실제 CSV에서 가져왔다.
모델 로딩은 약 84.525ms, 첫 cold 추론은 약 3729.502ms였고 반복 평균에서 제외했다.
결과는 한 번의 실행이며 시스템 부하에 따라 달라질 수 있다.

conf 0.10에서는 bus 1개·person 4개·skateboard 1개가 검출됐다. skateboard의 confidence는
약 0.1193으로 conf 0.25/0.60에서는 제외됐다. 나머지 5개 confidence는 모두 0.60을 초과했다.
이는 모델의 출력 비교이며 사람의 정답 판정은 아니다. 모든 bbox/class/confidence는
`artifacts/f101_conf_sweep.csv`에 저장했다.

산출물은 `measurement.csv` 3행, `f101_iterations.csv` 30행,
`f101_conf_sweep.csv` 3행이다. CSV에는 버전·장치·입력 크기·파일 SHA256이 포함된다.

## 오류와 조치, 남은 범위

명세서 PDF를 로컬에서 확인하기 위한 pypdf 설치가 처음에는 sandbox의 네트워크 제한
(WinError 10013)으로 실패했다. 원인을 확인하고 승인된 권한 확장으로 재시도해 6.21.1을 설치했다.
pypdf는 추론 런타임 필수가 아니며 `docs` 선택 의존성으로 분리했다.

현재 환경과 F101 구현·실행에 남은 차단 사항은 없다. VS Code의 실제 선택된 인터프리터 UI는
사용자가 `.venv`인지 확인해야 한다. 새 PC에서의 전체 재설치는 검증하지 않았고 현재 환경의
패키지 import, editable 설치 및 의존성 검사를 수행했다.
F102 이후는 미구현이며 VLM GGUF/mmproj의 로딩·추론 성공을 주장하지 않는다.
