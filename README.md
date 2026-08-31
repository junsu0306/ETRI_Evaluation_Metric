# ETRI Robot Compression Evaluation

Pittsburgh 장소 인식용 NetVLAD 모델을 기준으로 지식증류(KD), 필터 프루닝, 실제 채널 축소의 성능과 압축률을 평가하는 프로젝트입니다.

현재 정리된 기본 실험은 `ResNet18 teacher → MobileNetV2 student → pruning → structural reduction`입니다. 기존 실험 원본은 결과를 훼손하지 않도록 `trash/legacy`에 보존했습니다.

## 디렉터리

```text
.
├── src/robot_compression/   # 실행 코드
│   ├── data/                # Pittsburgh 로더와 hard-negative mining
│   ├── models/              # CNN encoder와 NetVLAD
│   ├── training/            # clustering, KD 학습, Recall 평가
│   └── compression/         # MobileNetV2 pruning/reduction
├── scripts/                 # 환경·자원·데이터 준비 도구
├── tests/                   # CPU 단위/smoke 테스트
├── docs/                    # 데이터 및 이전 진행 기록
├── artifacts/               # run, checkpoint, centroid, cache (Git 제외)
├── data/                    # 다운로드 파일과 데이터셋 (Git 제외)
└── trash/legacy/            # 이전 실험 코드 원본
```

작업 전 [RULE.md](RULE.md)를 반드시 확인합니다.

## 1. 독립 환경 준비

```bash
./scripts/bootstrap_env.sh
source .venv/bin/activate
```

환경은 Python 3.10, PyTorch 2.7.1 + CUDA 12.8 조합으로 고정했습니다. 시스템 Python에는 패키지를 설치하지 않습니다.

머신별 설정은 `.env`에 있습니다. 기본 물리 GPU는 2번이며 `CUDA_VISIBLE_DEVICES=2` 때문에 Python에서는 `cuda:0`으로 보입니다.

## 2. 서버 자원 확인

```bash
./scripts/check_resources.sh
```

공용 서버의 다른 프로세스는 종료하지 않습니다. GPU 2의 여유 메모리가 10 GiB 미만이면 학습을 시작하지 않습니다.

## 3. Pittsburgh 데이터 준비

```bash
source .env
./scripts/download_pittsburgh.sh
robot-compression check
```

기본 경로는 `data/datasets/pittsburgh`입니다. Pittsburgh 30k 재현에 필요한 약 6.6 GB의 압축 파일을 공개 배포 미러에서 받고 MD5를 검증한 다음 자동으로 압축을 풉니다. 중단 후 같은 명령을 실행하면 이어받습니다. 250k 전체 database image가 필요하면 `PITTSBURGH_FULL=1 ./scripts/download_pittsburgh.sh`를 사용합니다. 상세 내용은 [docs/DATASET.md](docs/DATASET.md)를 참고합니다.

## 4. 기본 검증

```bash
pytest
robot-compression smoke --cpu --arch mobilenetv2 --num-clusters 4 --pruning-ratio 0.2
```

GPU 검증은 자원 확인 후 실행합니다.

```bash
robot-compression smoke --arch mobilenetv2 --num-clusters 4 --pruning-ratio 0.2
```

## 5. 실험 순서

NetVLAD는 백본별 centroid가 필요하므로 학습보다 clustering을 먼저 실행합니다.

```bash
robot-compression cluster --arch mobilenetv2 --num-clusters 64

# teacher도 최초 1회 centroid 생성 후 일반 triplet loss로 학습
robot-compression cluster --arch resnet18 --num-clusters 64
robot-compression train \
  --arch resnet18 \
  --kd-weight 0 \
  --num-clusters 64

robot-compression train \
  --arch mobilenetv2 \
  --teacher-arch resnet18 \
  --teacher-checkpoint artifacts/teacher/resnet18_best.pt \
  --num-clusters 64 \
  --pruning \
  --pruning-ratio 0.15
```

평가와 실제 구조 축소는 다음과 같습니다.

```bash
robot-compression test \
  --checkpoint artifacts/runs/<run>/checkpoints/best.pt \
  --split test

robot-compression reduce \
  --checkpoint artifacts/runs/<run>/checkpoints/best.pt \
  --output artifacts/runs/<run>/checkpoints/best_reduced.pt
```

## 평가 지표

- 인식 성능: Recall@1, Recall@5, Recall@10, Recall@20
- 모델 규모: 전체 파라미터, 학습 가능 파라미터, non-zero 파라미터, 체크포인트 파일 크기
- 기존 기록의 78.3%는 `trainable parameters` 기준입니다. 전체 파라미터 기준으로는 약 36.1%이므로 공식 평가 기준을 확정해야 합니다.

## 현재 재현 범위

- NetVLAD forward, KD gradient, MobileNetV2 프루닝과 실제 구조 축소는 자동 테스트합니다.
- 전체 학습 결과를 재현하려면 Pittsburgh 데이터, backbone centroid, ResNet18 teacher checkpoint가 필요합니다.
- 이전 코드의 알려진 오류와 대응 내용은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 기록했습니다.
