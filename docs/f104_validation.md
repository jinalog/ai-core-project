# F104 YOLO11n 기본 파인튜닝 검증

2026-10-21, 기존 Python 3.12 가상환경에서 F104의 데이터 검증·10에포크 학습·동일 val 전후
평가를 실제 실행했다. F105·F106·광고 정책 RAG는 구현하지 않았다.
기존 F101~F103 코드·문서·측정 결과와 교육 원본은 수정하지 않았다.

## 데이터셋 구성과 원본 보호

기존 `source/2회차_day1/dataset/data.yaml`을 읽어 확인했다.
클래스는 `0: obstacle`, `1: marker`이고 모든 이미지는 320×320이다.

| split | 이미지 | 라벨 파일 | obstacle 박스 | marker 박스 | 배경 이미지 |
|---|---:|---:|---:|---:|---:|
| train | 40 | 40 | 28 | 30 | 2 |
| val | 10 | 10 | 10 | 6 | 0 |

총 50장·50개 라벨 파일, 74개 객체다. 빈 라벨은 객체 없는 배경 이미지로 허용했다.
이미지 디코딩, 이미지/라벨 짝, 중복 stem, 고아 라벨, 필드 수, 정수 클래스 ID,
유한한 0~1 좌표, 양수 폭/높이, bbox 경계 및 클래스 분포를 검사했다.
**오류 0개·경고 0개**였고 train/val 이미지 해시 중복도 없었다.
정규화 좌표의 반올림을 고려해 bbox 경계에는 1e-6 허용 오차를 사용한다.

이 데이터는 제공된 준비 스크립트가 사각형과 원을 그려 만든 **교육용 합성 도형 데이터**다.
실제 장애물·표식 사진이나 실제 운영 데이터셋이라고 주장하지 않는다.
기존 데이터가 있으므로 제공 스크립트의 `--demo`는 실행하지 않았다. 해당 스크립트는
`--demo --train-count 40 --val-count 10` 생성 옵션을 지원하지만 현재 교육 원본을 재생성하지 않는다.

Ultralytics는 cache=False에서도 라벨 `.cache`를 만들 수 있으므로 검증한 이미지·라벨만
`artifacts/f104/baseline/dataset/`에 **독립적인 비공개 작업 사본**으로 복사했다.
기존 라벨 캐시는 복사하지 않았고, 새 data.yaml의 path만 이 작업 폴더를 가리킨다.
모든 이미지·라벨의 SHA256과 원본/작업 사본 전체 digest가 일치함을 확인한 뒤 학습했다.
모델·사본·측정 CSV·캐시는 기존 `.gitignore`의 artifacts 제외 규칙으로 보호된다.

원본과 작업 사본의 데이터 digest:

```text
dataset: 25d4ef6148569fb8a8dbe1189513632e968e4dca770e86db3cb177ae2142d437
val:     6e31a982168f765acbb0d1548d886d0f30708ff3202125fe6f0825dee1e171b2
```

## 실행 방법

설치된 Ultralytics 및 그 의존성 PyYAML 6.0.3을 사용했다. 추가 패키지 설치는 하지 않았다.

```powershell
# 원본에 아무것도 쓰지 않는 데이터 검증
.\.venv\Scripts\python.exe -m experiments.f104_dataset_validate

# 데이터 검증 → 작업 사본 → 사전 평가 → 학습 → best.pt 사후 평가
.\.venv\Scripts\python.exe -u -m experiments.f104_finetune --epochs 10 --imgsz 320 --batch 8 --device cpu --seed 42

# 전체 회귀 및 F104 단위 테스트
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실측에서는 `--output-dir artifacts/f104/baseline`을 추가했다. 해당 폴더는 이미 존재하므로
재실행은 기본 실행별 폴더를 사용하거나 다른 새 `--output-dir`을 지정한다.
`--data`, `--weights`도 지원하지만 누락된 파일을 자동 생성·다운로드하지 않는다.
검증 실패·학습 오류·best.pt 누락은 비정상 종료와 execution.log의 예외 스택으로 남긴다.
실패해도 이미 실제 측정한 중간 산출물은 보존한다.

## 학습 환경과 하이퍼파라미터

실제 run_id는 `20261021T073213Z-d98712db`다. 2026-10-21 16:32(한국 시간)에 실행했다.

| 항목 | 실제 설정 |
|---|---|
| OS / Python | Windows 11 / 3.12.10 |
| Ultralytics / Torch | 8.4.164 / 2.14.0+cpu |
| 장치 | CPU, CUDA 사용 불가 |
| 시작 가중치 | 루트의 기존 `yolo11n.pt` |
| epochs / imgsz / batch | 10 / 320 / 8 |
| seed / workers | 42 / 0 |
| deterministic / CPU threads | True / 8 |
| optimizer 요청 / 실제 선택 | auto / AdamW |
| 실제 optimizer 초기 lr / momentum | 0.001667 / 0.9 |
| lrf / warmup_epochs | 0.01 / 3.0 |
| mosaic / close_mosaic | 1.0 / 10, 해당 10에포크 실험에서는 시작 시 mosaic 종료 |
| patience | 0, 지정한 10에포크를 모두 수행하도록 조기 종료 비활성화 |
| amp / cache / plots | False / False / False |
| pretrained / save | True / True |

epochs와 imgsz는 사용자 지정 조건이다. batch=8은 제공 학습 예제와 같은 값이고,
seed·workers·캐시 위치·early stop 비활성화 등은 이 구현의 재현/원본 보호를 위한 선택이다.
나머지는 버전 8.4.164의 실제 args.yaml과 training_args.json으로 보존했다.
args의 lr0=0.01은 auto optimizer에서 무시됐으며, 로그에 기록된 실제 AdamW lr은 0.001667이다.
제공 가중치의 448/499 항목이 전이됐고 모델 출력 클래스 수는 80에서 2로 바뀌었다.
라이브러리 설정 경로도 실행 폴더의 ultralytics_config로 지정했다.

학습 호출 전체 wall time은 **29.621초**였고 실제 10에포크가 완료됐다.
이 시간에는 train() 안의 검증·체크포인트 저장·최종 처리가 포함되며,
그 밖의 사전/사후 비교 평가와 작업 사본 생성은 제외된다.
학습 CSV의 마지막 epoch 누적 시간은 27.811초로 측정 구간이 다르다.

새 가중치:

```text
artifacts/f104/baseline/train/weights/best.pt
크기: 5,427,162 bytes
SHA256: 63c54d2ddf6bbd9820765349d58717393c8cd9ad377bd54250e95987c46d1df9
```

last.pt도 생성됐지만 전후 비교에는 라이브러리가 선정한 best.pt를 사용했다.
기존 `source/` 아래의 다른 학습 결과를 재사용하지 않았다.

### loss 기록

| loss | epoch 1 | epoch 10 |
|---|---:|---:|
| train box | 0.85306 | 0.63297 |
| train cls | 3.31146 | 1.21409 |
| train DFL | 0.99508 | 0.94521 |
| val box | 0.38503 | 0.64477 |
| val cls | 3.09867 | 2.87922 |
| val DFL | 0.78228 | 0.78566 |

train loss는 내려갔지만 모든 val loss가 개선된 것은 아니다. 이 표만으로 과적합 여부를
확정하지 않는다. epoch 10의 mAP50은 0.73185이며, 아래 best.pt의 지표와 혼동하지 않는다.
전체 10에포크 loss·lr·평가 지표는 training_history.csv와 train/results.csv에 보존했다.

## 동일 val 전후 비교

전후 모두 같은 val 이미지 10장과 라벨 16개를 사용했다.
각 모델의 평가 전후 val digest를 다시 확인했고 네 평가 행의 digest가 모두 같다.
설정은 imgsz=320, CPU, batch=8, workers=0, NMS IoU=0.7, max_det=300,
augment=False, rect=True다. conf=0.001과 0.25를 각각 평가했다.

| 모델 | conf | mAP50 | mAP50–95 | 고정 conf Precision | 고정 conf Recall | max_conf |
|---|---:|---:|---:|---:|---:|---:|
| before yolo11n.pt | 0.001 | 0.00000 | 0.00000 | 0.000000 | 0.00000 | 0.379307 |
| after best.pt | 0.001 | 0.79817 | 0.56064 | 0.007008 | 1.00000 | 0.039380 |
| before yolo11n.pt | 0.25 | 0.00000 | 0.00000 | 0.000000 | 0.00000 | 0.379307 |
| after best.pt | 0.25 | 0.00000 | 0.00000 | 0.000000 | 0.00000 | 0.039380 |

고정 conf P/R는 다음 방식으로 직접 측정했다. conf=0.001에서 얻은 val 예측을 해당 임계값으로
필터링하고, confidence 내림차순·동일 클래스·IoU≥0.5 조건으로 정답과 1:1 greedy 매칭했다.
전체 val의 TP/FP/FN을 합산하는 micro P/R이다. 원본 예측·정답 bbox도 평가 폴더에 저장했다.

| 모델 / conf | TP | FP | FN |
|---|---:|---:|---:|
| before / 0.001 | 0 | 26 | 16 |
| after / 0.001 | 16 | 2267 | 0 |
| before / 0.25 | 0 | 1 | 16 |
| after / 0.25 | 0 | 0 | 16 |

공식 val()의 mAP는 해당 conf로 구성한 후보 풀에 대한 AP다. 라이브러리의 box.mp/mr는
고정 conf에서의 P/R와 동일하다고 가정하지 않고 별도 컬럼으로 보존했다.
after/conf=0.001의 라이브러리 P/R 요약은 각각 0.008880 / 1.000000이다.
직접 계산한 P/R와 val()은 prediction 모드와 validation 모드의 후보 처리·평균 방식이 달라
동일한 수치일 필요가 없다. 참고: [공식 validation 문서](https://docs.ultralytics.com/modes/val/),
[metrics API](https://docs.ultralytics.com/reference/utils/metrics/).

max_conf는 운영 임계값에서 잘라낸 후의 값이 아니라 **conf=0.001 탐색의 모든 예측 중 최대값**이다.
따라서 동일 모델의 두 conf 행에서 같은 값이다. max_conf_probe_threshold도 CSV에 명시했다.
before의 max_conf=0.379307은 COCO의 다른 클래스에 대한 오검출이다.
원시 ID 0·1만의 최대값은 before 0.0, after 0.039380이며 target_id_max_conf로 별도 기록했다.

주의: 사전학습 모델의 ID 0·1은 person/bicycle, fine-tuned 모델의 ID 0·1은 obstacle/marker다.
사전학습 클래스를 임의로 재명명하거나 정답에 맞춰 변환하지 않았다.
before mAP=0은 이 도메인/클래스 헤드가 아직 정렬되지 않은 기준선이며,
이 비교만으로 동일 클래스 체계에서의 범용 정확도 개선을 입증했다고 주장하지 않는다.

## 해석과 한계, F105 개선 과제

낮은 후보 임계값에서는 mAP50이 0→0.79817로 높아졌지만, 최고 confidence는 0.03938로
0.25보다 낮다. **운영 conf=0.25에서는 정답 16개를 모두 놓치고 검출이 0개**였다.
낮은 conf에서 recall=1이어도 FP가 2267개이므로 사용 가능한 검출기가 됐다고 판단하지 않는다.
순위 기반 AP와 confidence 보정·고정 임계값 성능은 서로 다른 관점이다.

50장의 단순 합성 도형과 고정된 색·모양에 대한 1회 실험이며 실제 사진의 일반화 성능은
측정하지 않았다. val 10장은 표본이 작고 confidence가 낮은 원인도 이 실험만으로 확정하지 않는다.
train loss 감소를 운영 성능 통과로 대체하지 않는다. 추가 학습이나 하이퍼파라미터 탐색은 하지 않았다.

F105에서 검토할 과제이며 **이번에 구현하지 않은 내용**:

- 실제 운용 환경의 장애물·표식 각 100장 이상과 배경 이미지를 직접 촬영·라벨링한다.
- 조명·각도·거리 다양성을 확보하고 장면 단위로 train/val을 분리해 누수를 방지한다.
- 오검출·미검출 및 클래스별 confidence를 점검한 뒤 epochs와 imgsz를 한 번에 하나씩 변경한다.
- 같은 held-out val에서 conf=0.25의 P/R·max_conf를 재검증하고 운영 기준을 충족하는지 확인한다.

## 테스트와 보존 검증

기존 F101~F103 28개 + F104 9개 = **37개 통과**, 실제 실행 1.290초.
F104는 읽기 전용 형식/분포 검증, 잘못된 클래스·좌표·크기·숫자, 누락/고아/손상/장수 오류,
작업 사본 해시와 원본 독립성, 검증 후 데이터 변경 감지, 고정 conf의 클래스별 1:1 매칭,
best.pt 누락/빈 파일 감지, 동일 val과 평가 설정 및 CSV 출력을 검사했다.
단위 테스트의 작은 합성 JPEG·가짜 가중치/모델은 임시 검증 전용이며 실제 학습 성능과 구분했다.
실제 학습의 best.pt 생성·10에포크 완료·동일 val 평가는 위 실행에서 별도로 확인했다.

기존 Git 추적 파일 전체, 교육 source 파일 전체, 기존 측정 CSV/JSON 및 루트 yolo11n.pt,
총 **282개 파일**의 SHA256을 작업 전후 비교해 변경이 없음을 확인했다.
실행 중 변경될 수 있는 서버 로그·가상환경 전체는 비교 목록에 포함하지 않았다.
이번 작업은 새 코드·테스트·본 문서만 추가하며 기존 README와 F101~F103 문서도 보존했다.

## 산출물 위치

- `artifacts/f104/dataset_validation.json`: 읽기 전용 원본 검증.
- `artifacts/f104/baseline/run.json`, `execution.log`: 상태·실측 시간·환경·모델/dataset 해시.
- `original_validation.json`, `staged_validation.json`: 원본/작업 사본 검증 및 파일 해시.
- `training_args.json`, `training_history.csv`, `train/args.yaml`, `train/results.csv`: 전체 학습 설정·loss·지표.
- `train/weights/best.pt`, `last.pt`: 실제 새 학습 가중치.
- `finetune_compare.csv`: 전후 × conf 2개, 총 4행.
- `evaluation/*_val_predictions.json`: 실제 val 예측과 정답 bbox.
- `artifacts/f104/unit_tests.log`, `train_stdout.log`, `train_stderr.log`: 실제 테스트·학습 로그.
- `artifacts/f104/preservation_before.json`, `validation_summary.json`: 기존 파일 보존과 최종 산출물 검증.

학습·평가 실행에 남은 환경 차단 사항은 없다. conf=0.25 무검출과 실제 환경 검증 부재는
미해결 품질 제한으로 남겼다. 기존 PoC API의 YOLO 가중치 경로를 best.pt로 바꾸지는 않았다.
