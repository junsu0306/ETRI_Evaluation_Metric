# ETRI Robot Compression Evaluation

Pittsburgh30k 장소 인식용 `CNN encoder + NetVLAD` 모델의 pruning, 실제 구조
축소, Recall과 파라미터 압축률을 평가하는 프로젝트입니다.

현재 재현 파이프라인은 다음과 같습니다.

```text
ResNet18 baseline
        ↓ architecture replacement / legacy-comparable triplet training
MobileNetV2 (ImageNet V1, 320 channels) + NetVLAD (64 clusters)
        ↓ 26% last-projection filter pruning
83 zero channels
        ↓ structural reduction
MobileNetV2 (237 channels) + reduced NetVLAD input/centroids
```

## 최종 결과

물리 GPU 0에서 MobileNetV2를 26% pruning 설정으로 30 epoch 학습하고, 실제
237채널 구조로 축소한 뒤 Pittsburgh30k val/test를 다시 평가했습니다.

`int(320 × 0.26) = 83`이므로 실제 제거율은 83/320, 즉 **25.9375%**입니다.

| 모델 | 선택 기준 | Epoch | Test R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|---:|
| 26% zero-pruned | val R@5 | 2 | 0.6931 | 0.8461 | 0.8906 | 0.9244 |
| 26% structurally reduced | val R@5 | 2 | 0.6891 | **0.8417** | 0.8879 | 0.9222 |
| 26% zero-pruned | val R@1 | 4 | 0.6979 | 0.8445 | 0.8928 | 0.9274 |
| 26% structurally reduced | val R@1 | 4 | **0.6920** | 0.8392 | 0.8906 | 0.9250 |
| 26% structurally reduced | 마지막 | 30 | 0.6401 | 0.8201 | 0.8693 | 0.9089 |

학습 loss는 계속 감소했지만 validation 최고점은 R@5가 epoch 2, R@1이 epoch
4였습니다. 따라서 배포 시 R@1 우선이면 `best_r1_reduced.pt`, 기존 R@5 선택
기준을 따르면 `best_reduced.pt`를 사용해야 합니다. epoch 30 모델은 학습 완료
확인용으로 보존하지만 최종 성능 모델로 사용하지 않습니다.

### 압축률

| 모델 | Encoder ch | Descriptor dim | Total params | Trainable params |
|---|---:|---:|---:|---:|
| ResNet18 기준 모델 | 256 | 16,384 | 2,815,552 | 2,132,480 |
| MobileNetV2 zero-pruned | 320 | 20,480 | 1,852,672 | 514,880 |
| MobileNetV2 15% reduced | 272 | 17,408 | 1,800,352 | 462,560 |
| MobileNetV2 26% reduced | 237 | 15,168 | 1,762,202 | **424,410** |

기존 평가 방식인 학습 가능 파라미터 기준 압축률은 다음과 같습니다.

```text
100 × (2,132,480 - 424,410) / 2,132,480 = 80.0978%
```

- 학습 가능 파라미터 기준: **80.0978%**, 목표 80% 달성
- 전체 파라미터 기준: 37.4118%
- MobileNetV2 자체 대비 trainable parameter 감소: 17.5711%
- descriptor 차원 감소: 25.9375%

80% 달성을 위한 최소 제거량은 82/320채널, 즉 25.625%입니다. 81채널만
제거하면 학습 가능 파라미터가 목표 상한 426,496개보다 94개 많습니다.

15% R@5-best reduced 모델과 비교하면 26% 모델은 test R@1이 1.48%p, R@5가
0.57%p 낮습니다. 압축률 목표를 넘기는 대신 소폭의 Recall 손실이 발생했습니다.

전체 30 epoch 곡선, 구버전 결과, ImageNet V1/V2 진단과 축소 전후 비교는
[실험 종합 보고서](docs/EXPERIMENT_SUMMARY.md)에 정리했습니다.

## 프로젝트 구조

```text
.
├── src/robot_compression/
│   ├── compression/       # MobileNetV2 channel pruning/reduction
│   ├── data/              # Pittsburgh loader와 hard-negative mining
│   ├── models/            # CNN encoder와 NetVLAD
│   ├── training/          # clustering, 학습, Recall 평가
│   ├── cli.py             # check/cluster/train/test/reduce CLI
│   └── config.py          # 환경 경로 설정
├── scripts/               # 환경·데이터·자원 점검 스크립트
├── tests/                 # CPU 단위 및 smoke 테스트
├── docs/                  # 구조, 데이터, 실험 보고서
├── artifacts/             # checkpoint, centroid, cache, 로그(Git 제외)
├── data/                  # Pittsburgh 데이터(Git 제외)
└── trash/legacy/          # 옮겨온 구버전 코드
```

## 환경 준비

프로젝트 전용 `.venv`만 사용합니다. 시스템 Python이나 다른 프로젝트의 환경에는
패키지를 설치하지 않습니다.

```bash
./scripts/bootstrap_env.sh
source .venv/bin/activate
```

고정 환경은 Python 3.10, PyTorch 2.7.1 + CUDA 12.8입니다. 머신별 경로와 GPU
설정은 Git에서 제외되는 `.env`에 저장되며 예시는 `.env.example`에 있습니다.

기본 물리 GPU는 2번입니다. `CUDA_VISIBLE_DEVICES=2`로 실행되므로 Python
프로세스에서는 `cuda:0`으로 표시됩니다. 2026-09-01의 26% 실험만 사용자의
요청에 따라 물리 GPU 0을 명시적으로 사용했습니다.

## 공용 서버 자원 확인

다운로드, clustering, 학습, 대규모 평가 전에 실행합니다.

```bash
./scripts/check_resources.sh
```

- 다른 사용자의 프로세스를 종료하지 않습니다.
- 기본 GPU 2의 여유 메모리가 10 GiB 미만이면 새 학습을 시작하지 않습니다.
- 시스템 가용 RAM 64 GiB, 작업 디스크 100 GiB 이상을 확인합니다.

## Pittsburgh30k 준비

```bash
source .env
./scripts/download_pittsburgh.sh
robot-compression check
```

기본 경로는 `data/datasets/pittsburgh`입니다. 스크립트는 공개 배포 아카이브를
이어받기 방식으로 내려받고 MD5 검증 후 압축을 풉니다. 전체 Pittsburgh250k
database가 필요하면 다음처럼 실행합니다.

```bash
PITTSBURGH_FULL=1 ./scripts/download_pittsburgh.sh
```

상세 데이터 구조는 [데이터셋 문서](docs/DATASET.md)를 참고합니다.

## 기본 검증

```bash
pytest
robot-compression check
robot-compression smoke \
  --cpu \
  --arch mobilenetv2 \
  --num-clusters 4 \
  --pruning-ratio 0.26
```

GPU smoke test는 자원 확인 후 `--cpu`만 제거해 실행합니다.

## 실험 실행

### 1. NetVLAD centroid 생성

NetVLAD centroid는 backbone의 descriptor 공간에 종속되므로 architecture마다
별도로 생성해야 합니다.

```bash
robot-compression cluster --arch resnet18 --num-clusters 64
robot-compression cluster --arch mobilenetv2 --num-clusters 64
```

구버전 torchvision의 `pretrained=True`를 재현하기 위해 MobileNetV2는
`IMAGENET1K_V1`을 명시적으로 사용합니다. centroid도
`mobilenetv2_imagenet1k_v1_...hdf5` 이름으로 분리됩니다.

### 2. ResNet18 기준 모델 학습

```bash
robot-compression train \
  --arch resnet18 \
  --num-clusters 64 \
  --kd-weight 0 \
  --epochs 30
```

### 3. MobileNetV2 26% pruning, 30 epoch

이번 결과를 GPU 0에서 재현하는 명령입니다. 기본 규칙대로 GPU 2를 사용하려면
`CUDA_VISIBLE_DEVICES=0` 부분을 제거합니다.

```bash
CUDA_VISIBLE_DEVICES=0 robot-compression train \
  --arch mobilenetv2 \
  --num-clusters 64 \
  --epochs 30 \
  --batch-size 1 \
  --cache-batch-size 48 \
  --workers 8 \
  --kd-weight 0 \
  --pruning \
  --pruning-ratio 0.26
```

각 run에는 다음 파일이 생성됩니다.

- `metrics.csv`: epoch별 loss, learning rate, Recall, 파라미터 수
- `checkpoints/best.pt`: validation R@5 최고 모델
- `checkpoints/best_r1.pt`: validation R@1 최고 모델
- `checkpoints/latest.pt`: 마지막 epoch와 optimizer/scheduler 상태
- `run_metadata.json`: 실행 명령, Git 상태, 환경, GPU, seed

중단된 학습은 같은 전체 epoch 수와 `latest.pt`를 지정해 이어서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 robot-compression train \
  --arch mobilenetv2 \
  --num-clusters 64 \
  --epochs 30 \
  --batch-size 1 \
  --cache-batch-size 48 \
  --workers 8 \
  --kd-weight 0 \
  --pruning \
  --pruning-ratio 0.26 \
  --checkpoint artifacts/runs/<run>/checkpoints/latest.pt
```

### 4. 실제 구조 축소

```bash
robot-compression reduce \
  --checkpoint artifacts/runs/<run>/checkpoints/best_r1.pt \
  --output artifacts/runs/<run>/checkpoints/best_r1_reduced.pt \
  --pruning-ratio 0.26
```

`reduce`는 MobileNetV2 마지막 projection과 BatchNorm뿐 아니라 연결된 NetVLAD
assignment convolution과 centroid의 입력 차원도 함께 줄입니다. reduced
checkpoint는 학습을 재개할 수 없으므로 원본 optimizer/scheduler 상태를 포함하지
않습니다. 이번 reduced 파일 크기는 약 7.28 MB입니다.

### 5. Validation/Test 평가

```bash
robot-compression test \
  --checkpoint artifacts/runs/<run>/checkpoints/best_r1_reduced.pt \
  --split val \
  --cache-batch-size 48 \
  --workers 8

robot-compression test \
  --checkpoint artifacts/runs/<run>/checkpoints/best_r1_reduced.pt \
  --split test \
  --cache-batch-size 48 \
  --workers 8
```

## KD 관련 주의사항

옮겨온 구버전 코드는 student soft-assignment logits까지 `no_grad` 안에서 계산해
실제 KD gradient가 없었습니다. 또한 teacher와 student가 독립적으로 만든
K-means cluster 번호는 서로 정렬되어 있지 않습니다.

현재 코드는 student KD gradient가 흐르도록 수정했지만 cluster 정렬 문제는 별도
해결이 필요합니다. 따라서 구버전 성능과 비교하는 공식 pruning 실험은
`--kd-weight 0`, 즉 triplet loss만 사용합니다. 양수 `--kd-weight` 결과를 구버전
KD 재현 결과로 해석하면 안 됩니다.

## 문서와 산출물

- [실험 종합 보고서](docs/EXPERIMENT_SUMMARY.md)
- [15% GPU 0 재현 기록](docs/REPRODUCTION_GPU0_20260831.md)
- [아키텍처와 구버전 대응](docs/ARCHITECTURE.md)
- [Pittsburgh 데이터셋](docs/DATASET.md)
- [옮겨온 구버전 진행 기록](docs/PROGRESS_LEGACY.md)

학습 checkpoint, centroid, TensorBoard 로그와 데이터셋은 용량 및 재생성 가능성을
고려해 Git에서 제외됩니다. 26% run의 로컬 원시 결과는 다음 위치에 있습니다.

```text
artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/
├── metrics.csv
├── evaluation.json
├── run_metadata.json
└── checkpoints/
    ├── best_reduced.pt
    ├── best_r1_reduced.pt
    └── latest_reduced.pt
```

코드 변경 검증 결과는 `pytest` 5개 통과, Python 컴파일 통과, CLI dataset check
통과, CPU pruning/reduction/KD-gradient smoke test 통과입니다.
