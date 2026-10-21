# 재현 안내

모든 명령은 저장소 루트와 Python 3.10 `.venv`에서 실행한다.
공개 저장소에는 데이터, 학습 checkpoint, centroid, cache가 포함되지 않는다.
아래는 같은 방법으로 새 실험을 수행하는 절차다. 보고된 checkpoint를 그대로 재평가하려면
별도 보관된 가중치가 필요하며 SHA-256은 [provenance.json](../results/provenance.json)에 있다.

## 1. 환경과 데이터

```bash
./scripts/bootstrap_env.sh
cp .env.example .env
```

Python 실행 파일명이 다르면 `PYTHON_BIN=/path/to/python3.10 ./scripts/bootstrap_env.sh`로 지정한다.
`.env`에서 사용 가능한 물리 GPU 번호를 설정한다. 기본 예시는 2이며 프로그램에서는 `cuda:0`이다.
원래 실험은 RTX A5000, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 환경이었다.
GPU가 없는 환경에서도 단위 테스트·CPU smoke·저장된 결과 검증은 가능하다.

```bash
./scripts/check_resources.sh
./scripts/download_pittsburgh.sh
.venv/bin/python -m robot_compression.cli check
```

다운로드·학습·평가 전 자원을 확인한다. 실험 운영 기준은 GPU 여유 메모리 10 GiB,
가용 RAM 64 GiB, 디스크 여유 100 GiB 이상이며 query batch는 1부터 시작한다.
다른 사용자 프로세스를 종료하지 않는다. 데이터 경로와 split은 [데이터셋 문서](DATASET.md)를 따른다.

## 2. Backbone별 centroid 생성

```bash
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli cluster \
  --arch resnet18 --num-clusters 64 --seed 42 \
  --num-descriptors 50000 --descriptors-per-image 100 --cache-batch-size 48 --workers 8
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli cluster \
  --arch mobilenetv3-large --num-clusters 64 --seed 42 \
  --num-descriptors 50000 --descriptors-per-image 100 --cache-batch-size 48 --workers 8
```

ImageNet 초기 가중치는 torchvision이 다운로드하고, centroid는 `artifacts/centroids/`에 저장한다.
원래 최종 실험은 미리 생성된 centroid를 재사용했다. 새 centroid를 생성한 실험은 입력 가중치가
달라질 수 있으므로 기존 수치와의 byte 단위 재현을 보장하지 않는다.

## 3. Teacher 학습과 선택

원래 teacher는 상한 50 epoch / R@1 patience 10 설정에서 19 epoch 완료 후 수동 종료했다.
그중 validation R@1 최고인 epoch 14를 공식 teacher로 사용했다.
아래 명령은 같은 완료 구간을 실행하도록 상한을 19로 제한한다. 새 실행에서도 선택은 validation만 사용한다.

```bash
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli train \
  --arch resnet18 --num-clusters 64 --kd-weight 0 \
  --epochs 19 --patience 10 --early-stop-rank 1 \
  --batch-size 1 --cache-batch-size 48 --workers 8 \
  --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.001 \
  --margin 0.1 --seed 42 --run-dir artifacts/runs/reproduce_teacher
```

Student에 넘길 파일은 `artifacts/runs/reproduce_teacher/checkpoints/best_r1.pt`다.
보고된 공식 teacher SHA는 `7340574ae75d08e07fd324e82d2932842205685c8edcd4a8719cc16309f3f897`이다.
새 학습의 최고 epoch와 SHA가 같다는 보장은 없다. 원래 설정은
[teacher 설정 JSON](../results/configs/teacher_epoch14_official.json)에 보존했다.

## 4. V3 72/76/80% KD-on

아래 shell 변수는 같은 Bash 세션에서 사용한다. 공통 설정은 모든 학생에서 유지한다.

```bash
teacher_checkpoint=artifacts/runs/reproduce_teacher/checkpoints/best_r1.pt
student_options=(
  --arch mobilenetv3-large --teacher-arch resnet18
  --teacher-checkpoint "$teacher_checkpoint" --num-clusters 64
  --epochs 50 --patience 10 --early-stop-rank 1
  --batch-size 1 --cache-batch-size 48 --workers 8
  --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.001
  --margin 0.1 --seed 42 --temperature 0.1 --pruning
)

./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli train "${student_options[@]}" \
  --kd-weight 0.5 --pruning-ratio 0.27708333333333335 \
  --run-dir artifacts/runs/reproduce_v3_c72
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli train "${student_options[@]}" \
  --kd-weight 0.5 --pruning-ratio 0.384375 \
  --run-dir artifacts/runs/reproduce_v3_c76
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli train "${student_options[@]}" \
  --kd-weight 0.5 --pruning-ratio 0.49166666666666664 \
  --run-dir artifacts/runs/reproduce_v3_c80
```

각 run 디렉터리는 새 경로여야 한다. 첫 KD-on 학습에서 teacher descriptor cache를 만들고
후속 학생은 동일 checkpoint hash의 cache를 재사용한다. 명령은 순차 실행되며 자동 병렬 작업을 시작하지 않는다.
`--pruning-ratio`는 expansion 채널 제거 비율이다. 목표 trainable 압축률 0.72/0.76/0.80을 직접 넣지 않는다.

## 5. 80% KD-off 대조군

동일한 `student_options`와 teacher를 사용한다. 일반 CLI의 KD weight만 0으로 바꾸면 triplet 계수가
1이 되므로, 최종 대조실험을 재현할 때는 전용 진입점을 사용한다.

```bash
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.kd_control train "${student_options[@]}" \
  --kd-weight 0 --pruning-ratio 0.49166666666666664 \
  --run-dir artifacts/runs/reproduce_v3_c80_off
```

이 모듈은 fresh V3 80% 실험만 지원하며 `0.5 × Triplet`을 적용한다.
CSV의 `triplet_loss`는 가중 전 값, `loss`는 가중 후 값이고 `kd_loss`는 0이다.
Teacher 생성의 난수 소비를 유지하되 실제 teacher 추론은 하지 않는다.

## 6. 구조 축소와 평가

아래를 각 학생 run에 대해 실행한다. `best_r1.pt`는 validation으로 선택된 원본 zero-pruned 모델이다.
구조 축소 때 `--pruning-ratio 0`을 지정해 추가 채널 선택 없이 기존 zero 채널만 제거한다.

```bash
student_run=artifacts/runs/reproduce_v3_c80
.venv/bin/python -m robot_compression.cli reduce --cpu \
  --checkpoint "$student_run/checkpoints/best_r1.pt" --pruning-ratio 0 \
  --output "$student_run/checkpoints/best_r1_reduced.pt"

./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli test \
  --checkpoint "$student_run/checkpoints/best_r1.pt" --split test \
  --cache-batch-size 48 --workers 8 --output "$student_run/zero_test.json"
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli test \
  --checkpoint "$student_run/checkpoints/best_r1_reduced.pt" --split val \
  --cache-batch-size 48 --workers 8 --output "$student_run/reduced_val.json"
./scripts/check_resources.sh
.venv/bin/python -m robot_compression.cli test \
  --checkpoint "$student_run/checkpoints/best_r1_reduced.pt" --split test \
  --cache-batch-size 48 --workers 8 --output "$student_run/reduced_test.json"
```

Teacher test도 같은 `test` 명령에 teacher의 `best_r1.pt`를 전달한다.
Zero-pruned validation은 선택 epoch의 `metrics.csv` 기록을 사용한다.
Reduced checkpoint는 추론용이며 학습 재개에 사용하지 않는다.

## 7. 결과 확인

```bash
.venv/bin/python scripts/verify_results.py
```

이 명령은 공개된 기존 결과 파일을 검사한다. 새 실행은 `artifacts/runs/`에 별도로 기록되고
공개 `results/`를 자동으로 덮어쓰지 않는다. 새 실험은 seed·환경·checkpoint 선택을 함께 기록해 비교한다.
