# 프로젝트 작업 규칙

## 1. 독립 환경

- 모든 Python 명령은 저장소의 `.venv`에서 실행한다.
- 시스템 Python에 `pip install`하지 않는다. 설치는 `.venv/bin/python -m pip`만 사용한다.
- 의존성을 변경하면 `requirements.txt`의 버전을 고정하고 환경 점검을 다시 수행한다.
- 머신별 경로와 GPU 설정은 `.env`에 두고 Git에는 `.env.example`만 기록한다.

## 2. 공용 서버 자원

- 다운로드, 학습, 평가 전 `scripts/check_resources.sh`를 실행한다.
- 기본 물리 GPU는 **2번**이다. `.env`의 `CUDA_VISIBLE_DEVICES=2`를 사용하므로 프로그램 내부 장치명은 `cuda:0`이다.
- 다른 사용자의 프로세스를 종료하거나 GPU 메모리를 강제로 정리하지 않는다.
- GPU 2번의 여유 메모리가 10 GiB 미만이면 새 학습을 시작하지 않는다. 사용자가 다른 GPU 사용을 명시적으로 승인하거나 GPU가 빌 때까지 기다린다.
- 시스템 가용 RAM이 64 GiB 미만이거나 작업 디스크 여유가 100 GiB 미만이면 대규모 작업을 시작하지 않는다.
- 배치 크기는 1에서 시작해 실제 최대 GPU 메모리를 확인한 뒤 올린다.

## 3. 데이터와 산출물

- Pittsburgh 데이터는 `data/datasets/pittsburgh`에만 둔다.
- 원본 아카이브는 `data/downloads/pittsburgh`, 모델·로그·캐시는 `artifacts/`에 둔다.
- 데이터, 체크포인트, 캐시를 Git에 커밋하지 않는다.
- 데이터 다운로드는 `scripts/download_pittsburgh.sh`만 사용하고 MD5 검증 후 압축을 푼다.
- 삭제보다 `trash/` 이동을 우선한다. `trash/`의 파일을 영구 삭제하려면 사용자 확인을 받는다.

## 4. 실험 재현성

- 실행 명령, Git commit, 환경 버전, 데이터 경로, GPU, seed를 run 디렉터리에 기록한다.
- NetVLAD 학습 전 해당 백본·데이터셋·cluster 수의 centroid를 먼저 생성한다.
- 압축률은 `total parameters`, `trainable parameters`, 직렬화 파일 크기를 모두 기록한다. 공식 지표로 어느 값을 사용할지는 별도로 명시한다.
- KD에서는 teacher만 `no_grad`로 계산하고 student logits에는 gradient가 흐르는지 테스트한다.
- 프루닝은 명시적인 `--pruning` 옵션이 있을 때만 요청 비율로 수행한다.

## 5. 코드 품질

- 실행 코드는 `src/robot_compression`, 실행 보조 도구는 `scripts`, 검증은 `tests`에 둔다.
- 구버전·중복 실험 코드는 수정하지 않고 `trash/legacy`에 보존한다.
- 변경 후 최소한 `pytest`, CLI `check`, CPU smoke test를 실행한다. GPU 작업 전에는 자원 점검을 다시 한다.
