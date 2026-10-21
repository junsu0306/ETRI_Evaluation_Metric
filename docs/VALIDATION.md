# 공개 버전 검증

2026-10-21 공개 패키지 검증 기록이다. 정확도 수치는 2026-09-11 완료한 원본 실험에서 가져왔다.
이 작업에서 전체 데이터셋 재학습이나 GPU 정확도 평가는 수행하지 않았다.

## 실행 검증

`scripts/check_resources.sh`로 자원을 확인한 뒤 아래 Python 검증에 공통으로
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''`를 적용했다.

| 검사 | 결과 |
|---|---|
| `.venv/bin/python -m pytest --tb=short` | **24 passed**, 5.78초, 기존 SWIG deprecation warning 3개 |
| `.venv/bin/python -m robot_compression.cli check` | dataset ready, validation DB 10,000/query 7,608 |
| V3-Large/NetVLAD64 CPU smoke, ratio=472/960 | forward와 reduced 출력 `(1,10240)`, 유지 채널 488, reduced total 2,445,152 |
| CPU smoke KD gradient | norm 0.00000736, student gradient 정상 |
| `.venv/bin/python scripts/verify_results.py` | 공개 파일 checksum 38개, 비교 14행, 전체 epoch 기록 8개 확인 |
| Bash 문법 검사 | bootstrap/download/resource 스크립트 통과 |
| `git diff --cached --check` | 통과 |

GPU를 숨겼으므로 CLI check의 CUDA unavailable은 의도된 상태다.
CPU smoke는 모든 파라미터를 학습 가능하게 생성한다. 이때 출력되는 trainable 수는
앞부분을 동결한 실제 학습 모델의 공식 압축률 계산에 사용하지 않는다.

## 공개 파일만으로 재검증

Git index에 있는 파일만 임시 디렉터리로 추출했다. 로컬 `.env`, 데이터, checkpoint,
cache, 과거 controller 없이 해당 디렉터리의 `src/`만 `PYTHONPATH`로 지정했다.

- 기존 `.venv` 의존성으로 테스트 **24 passed**, 6.62초, 동일 SWIG warning 3개.
- CPU smoke 성공: 출력 `(1,10240)`, 유지 채널 488, KD gradient norm 0.00000736.
- 결과 검증 스크립트 성공, KD-on − KD-off Test R@1 **+2.6262%p** 확인.
- 데이터 로더가 실제로 공개 추출본의 `src/robot_compression/data/pittsburgh.py`에서 import됨을 확인.

새 환경에 대한 전체 dependency 재설치는 수행하지 않았다. 기준 `.venv`에서 공개 소스의
독립성을 검증했다. 설치 스크립트에 Python 3.10 확인을 추가하고 Bash 문법 검사를 수행했다.

## 결과와 Git 추적 범위

- 공개 파일은 코드·테스트·문서·설정·결과 **70개**이며, `results/`에는 CSV/JSON 39개를 포함한다.
- 원본 checkpoint 14개 형태의 SHA-256과 원본 평가 JSON의 epoch/split/Recall을 대조했다.
- 공개 CSV는 수치를 유지하고 LF 줄바꿈으로 정규화했다. 로컬 절대 경로 접두사는 제거했다.
- 원본 파일과 공개 파일의 SHA를 [provenance.json](../results/provenance.json)에 구분해 기록했다.
- 공개 문서의 로컬 링크는 모두 Git 추적 파일을 가리키는지 검사했다.
- `.env`, `data/`, `artifacts/`, `trash/`, `.venv/`, 로컬 에이전트 설정은 Git 추적 대상에서 제외했다.
- 기존 로컬 산출물 310개의 크기·수정 시각이 유지됨을 확인했다. 과거 문서와 서버 전용 코드는 로컬 보관했다.
- 기존 Git 이력은 유지한다. `.gitignore`와 추적 해제는 현재 트리에 적용되며 과거 commit을 재작성하지 않는다.

본 공개 버전의 테스트는 KD gradient, 0.5×Triplet 대조군, teacher 추론 생략,
V3 구조 축소 전후 descriptor, R@1 최고 모델 선택·patience 경계를 포함한다.
서버 전용 campaign의 프로세스·실시간 보고서 테스트는 공개 실행 경로에서 제외했다.
