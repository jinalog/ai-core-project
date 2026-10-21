# F102 VLM 선정과 실행 설계

F102는 프로젝트 명세서 5~6쪽에 따라 Qwen2.5-VL-3B-Instruct GGUF Q4_K_M와
해당 모델의 mmproj를 사용한다. 라이브러리는 기존 llama-cpp-python 0.3.34다.
F101 코드·측정 CSV·교육 원본 소스는 수정하지 않는다.

## 선정 근거

기준 모델은 명세서가 지정한 3B 비전 언어 모델이며, 현재 CPU 환경에서 실행할 수 있는
GGUF 양자화 파일과 멀티모달 핸들러가 준비되어 있다. 모델 본체만으로는 이미지 입력을
처리할 수 없어 같은 모델 계열의 mmproj도 함께 로드한다.

| 본체 후보 | 공개 파일 크기 | 이번 선택 |
|---|---:|---|
| Q4_K_M | 약 1.93 GB | 명세서 지정, 로컬 파일 존재 |
| Q8_0 | 약 3.29 GB | 미다운로드·미실행 |
| F16 | 약 6.18 GB | 미다운로드·미실행 |

출처: [ggml-org GGUF 모델 카드](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF).
파일 크기는 실행 RAM과 다르며 mmproj·KV cache·비전 인코딩 메모리를 포함하지 않는다.
이번 로컬 파일은 본체 1,929,901,056바이트, mmproj 1,338,428,128바이트다.
양자화 후보 간 정확도·속도를 이 PC에서 비교하지 않았으므로 수치 우위를 주장하지 않는다.
샘플 한 장의 설명은 정량 정확도 평가를 대체하지 않는다.

라이선스 표시는 원본과 변환 저장소를 구분한다. 확인 시점에 GGUF 저장소 메타데이터에는
Apache-2.0이 표시되지만 [원본 모델 카드](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md)와
[원본 LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)는 Qwen Research License다.
변환 저장소의 표시만으로 원본의 사용 조건이 바뀌었다고 판단하지 않는다.
이번 작업은 로컬 교육·평가 실행이며 모델 파일을 저장소에 포함하지 않는다.

## 실제 경로

- 본체: `models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`
- mmproj: `models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf`
- 이미지: `source/2회차_day1/samples/sample.jpg`

경로는 기존 `Settings`를 재사용하며 `--model`, `--mmproj`, `--image`로 바꿀 수 있다.
세 파일의 존재·빈 파일·GGUF 매직과 JPEG 디코딩을 검사한다. 실제 모델 호환성은
라이브러리의 모델/비전 컨텍스트 초기화로 확인한다. 원본 JPEG 바이트를 data URL로 전달하며
복사·리사이즈·별도 이미지 생성·네트워크 이미지 요청을 하지 않는다.

## 명세서 이외의 설계 선택

- 비교 프롬프트: 한국어 한 문장(`ko_short`), 한국어 세 문장(`ko_detail`), 영어 한 문장(`en_short`).
- 각 프롬프트를 연속 2회 실행하고 모든 호출에 temperature=0.0, seed=42를 동일 적용한다.
  첫 비교는 max_tokens=160이었다. 상세 출력이 잘린 것을 확인해 기본 한도를 320으로 바꾸고
  상세 프롬프트를 동일 조건 2회로 별도 재검증했다. 기존 160토큰 결과는 변경하지 않는다.
- CPU 기본값 n_ctx=8192, n_threads=8. 이 설정은 F106 최적화 실험을 구현한 것이 아니다.
- 0.3.34의 Qwen25VLChatHandler는 호출 때마다 KV 및 입력 상태를 초기화하므로 대화 이력을
  누적하지 않는다. 같은 핸들러·모델 인스턴스에서 조건을 고정해 비교한다.
- 0.3.34는 mmproj를 지연 로드한다. `_init_mtmd_context()`를 먼저 호출해 실제 mmproj 로딩을
  추론에서 분리했다. 이 버전에는 핸들러 public close가 없어 `_exit_stack.close()`로 해제한다.
  두 private API 사용은 버전 0.3.34에 한정하며 별도 단위 테스트로 호출·실패 정리를 확인한다.
  참고: [해당 버전의 공식 핸들러 구현](https://github.com/abetlen/llama-cpp-python/blob/v0.3.34/llama_cpp/llama_chat_format.py).
- 로딩은 handler 구성, 본체, mmproj 각각의 wall time을 기록한다. 추론은 매 호출의
  이미지 디코딩·비전 인코딩·입력 평가·텍스트 생성을 포함한다. 이미지 base64 인코딩·파일 SHA256·CSV 쓰기는 제외한다.
- 응답 원문을 수정하지 않고 문자열 완전 일치로 재현성을 판정한다. `finish_reason=length`인
  동일 잘림 결과는 완성된 응답의 재현 성공으로 처리하지 않는다.
- 언어 표시는 한글 음절·라틴 문자 수에 따른 휴리스틱이다. 문자 수, 공백 기준 어절 수,
  실제 completion 토큰 수와 종료 사유를 함께 기록한다. 의미 정확도·요청 문장 수는 실제 출력으로 검토한다.
  토큰 수는 라이브러리 응답의 usage 값을 그대로 보존하며, 비전 토큰을 포함한 전체
  컨텍스트 사용량이라고 해석하지 않는다.
- 출력 폴더는 실행마다 새로 만들어 기존 결과를 덮어쓰지 않는다. 완료된 호출마다 CSV를
  갱신하고 실패 시 실행 상태·예외·스택을 남긴다. 실패 후 재시도는 다른 폴더를 사용한다.

검증 결과는 [F102 검증 기록](f102_validation.md)에 정리한다. F103 API는 구현하지 않는다.
