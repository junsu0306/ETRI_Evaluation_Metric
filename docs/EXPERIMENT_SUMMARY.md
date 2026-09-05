# 로봇 압축 실험 종합 보고서

최종 갱신: 2026-09-01

현재 코드의 pruning/reduction 구현 방법과 최종 결과를 함께 보려면
[`implmentation.md`](implmentation.md)를 우선 참고한다. 이 문서는 실험 진행
과정과 구버전 비교 기록을 보존한다.

## 1. 결론

MobileNetV2 마지막 projection에 `--pruning-ratio 0.26`을 적용하고 30 epoch를
학습한 뒤 실제 구조 축소까지 완료했다. 320개 출력 채널에서 정수 내림으로
83개가 제거되어 실제 채널 감소율은 **25.9375%**, 남은 채널은 237개다.

ResNet18 teacher의 학습 가능 파라미터 2,132,480개를 기준으로 하면 최종
MobileNetV2 + NetVLAD 모델의 학습 가능 파라미터는 424,410개이며, 다음 산식으로
목표 80%를 달성한다.

```text
100 × (2,132,480 - 424,410) / 2,132,480 = 80.0978%
```

다만 전체 파라미터 기준 압축률은 37.4118%다. 따라서 “80% 달성”은 기존 기록과
동일한 **학습 가능 파라미터 기준**일 때만 성립한다.

최종 배포 후보는 목적에 따라 두 개다.

| 목적 | 체크포인트 | Epoch | Test R@1 | Test R@5 |
|---|---|---:|---:|---:|
| R@1 우선 | `best_r1_reduced.pt` | 4 | **0.6920** | 0.8392 |
| R@5 우선 | `best_reduced.pt` | 2 | 0.6891 | **0.8417** |

epoch 30 모델의 test R@1은 구조 축소 후 0.6401까지 낮아졌다. 학습 loss는
감소했지만 validation recall이 초기 최고점에서 지속적으로 떨어졌으므로 마지막
모델보다 early checkpoint를 사용해야 한다.

## 2. 지표와 비교 기준

프로젝트가 사용해 온 최적화율은 다음과 같다.

```text
최적화율(%) = 100 × (기준 모델 B - 최적화 모델 A) / 기준 모델 B
```

파라미터 범위에 따라 값이 크게 달라지므로 아래 세 지표를 분리한다.

- 전체 파라미터: frozen backbone까지 포함한 모델의 모든 `Parameter` 수
- 학습 가능 파라미터: `requires_grad=True`인 `Parameter` 수
- 체크포인트 크기: 직렬화 형식과 optimizer 포함 여부에도 영향을 받는 파일 크기

공식 80% 판정에는 기존 보고서와 동일하게 ResNet18의 학습 가능 파라미터
2,132,480개를 B로 사용했다. 전체 파라미터 기준 B는 2,815,552개다.

80% 달성을 위한 모델 A의 상한은 426,496개다. 마지막 descriptor channel 한
개를 구조 축소할 때 현재 구조에서는 1,090개 파라미터가 제거되므로 81채널을
제거한 426,590개는 94개 초과하고, 최소 82채널을 제거해야 한다. 따라서 최소
비율은 `82 / 320 = 25.625%`다. 이번 옵션 0.26은 구현의
`int(320 × 0.26)`에 따라 83채널을 제거해 경계를 한 채널 여유 있게 넘는다.

인식 성능은 Pittsburgh database/query 검색의 Recall@1, @5, @10, @20이다.

## 3. 공통 환경

| 항목 | 설정 |
|---|---|
| 데이터 | Pittsburgh30k train/val/test |
| 데이터 경로 | `data/datasets/pittsburgh` |
| Train | database 10,000 / query 7,416, 학습 query 7,320 |
| Validation | database 10,000 / query 7,608 |
| Test | database 10,000 / query 6,816 |
| Pooling | NetVLAD, 64 clusters |
| MobileNetV2 초기 가중치 | torchvision `IMAGENET1K_V1` |
| MobileNetV2 centroid | `mobilenetv2_imagenet1k_v1_pitts30k_64_desc_cen.hdf5` |
| Optimizer | SGD, lr 1e-4, momentum 0.9, weight decay 1e-3 |
| Scheduler | StepLR, 5 epoch마다 lr × 0.5 |
| Triplet margin | `sqrt(0.1)` |
| Seed | 42 |
| Python 환경 | 저장소 전용 `.venv` |
| PyTorch / CUDA runtime | 2.7.1+cu128 / 12.8 |

이번 26% 실험은 사용자의 명시적 요청에 따라 물리 GPU 0을 사용했다.
`CUDA_VISIBLE_DEVICES=0`이므로 프로그램 내부에서는 `cuda:0`이다. 학습은
2026-09-01 10:56부터 20:15까지 진행됐으며, 공용 GPU의 다른 작업과 경합한
구간이 있어 총 벽시계 시간은 약 9시간 20분이었다. GPU 메모리 부족이나 학습
재개는 발생하지 않았다.

## 4. 구현 파이프라인

1. ImageNet V1 MobileNetV2의 classifier와 마지막 1×1 expansion을 제외해
   local descriptor encoder를 구성한다. 마지막 InvertedResidual projection의
   출력은 320채널이다.
2. 해당 encoder로 추출한 50,000개 descriptor에 Faiss K-means를 수행해
   MobileNetV2 전용 64개 NetVLAD centroid를 만든다.
3. Pittsburgh hard-negative mining으로 query/positive/negative triplet을 만들고
   triplet loss를 학습한다. 구버전 비교 run에서는 `kd_weight=0`을 사용한다.
4. 매 optimizer step 뒤 마지막 1×1 projection의 출력 filter L2 norm이 작은
   채널을 지정 비율만큼 0으로 만들고, 연결된 BatchNorm affine 값도 0으로
   유지한다.
5. `reduce`에서 0인 projection 채널을 실제로 제거한다. 동시에 BatchNorm,
   NetVLAD assignment convolution 입력 채널, NetVLAD centroid 차원도 같은
   index로 줄인다. NetVLAD cluster 수 64는 유지한다.
6. 축소된 state dict를 다시 로드해 forward 출력과 val/test Recall을 측정한다.

26% 설정에서는 projection weight `83 × 960 = 79,680`개와 BatchNorm affine
166개가 0이 되어 학습 checkpoint의 zero parameter가 79,846개다. 구조 축소 시
NetVLAD assignment weight와 centroid도 각각 `64 × 83 = 5,312`개씩 제거되므로
최종 제거 파라미터는 총 90,470개다.

NetVLAD를 별도로 cluster-pruning하지 않은 이유는 현재 실험의 pruning 단위가
encoder descriptor channel이기 때문이다. 대신 구조 축소 단계에서 그 채널을
소비하는 NetVLAD의 입력 축을 함께 줄인다. NetVLAD cluster 자체를 줄이는 것은
centroid 재학습과 별도의 품질 검증이 필요한 다음 실험 범위다.

## 5. 옮겨온 구버전 기록

아래 값은 기존 프로젝트 문서와 결과에서 옮겨온 기록이다. 현재 코드로 다시
측정한 값과 혼동하지 않도록 “구버전 기록”으로 구분한다.

| 모델 | Split | R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|
| ResNet18 baseline | val | 0.7805 | 0.9172 | 0.9451 | 0.9670 |
| ResNet18 baseline | test | 0.7399 | 0.8750 | 0.9117 | 0.9420 |
| ResNet18 마지막 블록 pruning | test | 0.6504 | 0.8047 | 0.8555 | 0.9004 |
| MobileNetV2 원본, 30 epoch | test | 0.7061 | 0.8521 | 0.8994 | 0.9330 |
| MobileNetV2 pruning | test | 0.7193 | 0.8688 | 0.9096 | 0.9398 |
| MobileNetV2 reducing | val | 0.7337 | 0.8933 | 0.9292 | 0.9579 |
| MobileNetV2 reducing | test | 0.7199 | 0.8662 | 0.9080 | 0.9384 |

구버전 reduced 모델은 학습 가능 파라미터 462,560개로 ResNet18 대비 78.3088%
압축이었다. 이 수치는 현재 구현의 15% 설정인 48/320채널 제거와 일치한다.

## 6. 현재 코드에서 수행한 실험

### 6.1 ResNet18 teacher 재현

run: `artifacts/runs/20260831_164217_resnet18_netvlad`

| Split | R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|
| val, epoch 4 | 0.7896 | 0.9114 | 0.9426 | 0.9661 |
| test, epoch 4 | 0.8078 | 0.9024 | 0.9300 | 0.9520 |

구버전 baseline 수준을 확인한 뒤 epoch 4에서 종료했다.

### 6.2 ImageNet V2 진단 run

처음에는 현재 torchvision의 MobileNetV2 `DEFAULT`, 즉
`IMAGENET1K_V2`를 사용했다. 저장된 best checkpoint를 이번 데이터 로더로 다시
평가한 결과는 다음과 같다.

| 실험 | Epoch | Split | R@1 | R@5 | R@10 | R@20 |
|---|---:|---|---:|---:|---:|---:|
| V2 + soft-assignment KD | 2 | val | 0.3785 | 0.5860 | 0.6722 | 0.7560 |
| V2 + soft-assignment KD | 2 | test | 0.4710 | 0.6898 | 0.7748 | 0.8438 |
| V2 + triplet only | 1 | val | 0.3822 | 0.5715 | 0.6555 | 0.7384 |
| V2 + triplet only | 1 | test | 0.4941 | 0.6935 | 0.7669 | 0.8339 |

- KD run: `artifacts/runs/20260831_174635_mobilenetv2_netvlad`
- triplet run: `artifacts/runs/20260831_182807_mobilenetv2_netvlad`

구버전 torchvision의 `pretrained=True`는 V1이었지만 현재 `DEFAULT`는 V2다.
V2 encoder에 구버전 계열 centroid를 사용한 것이 큰 성능 저하의 주원인으로
판단되어 두 run은 진단 후 중단했다.

### 6.3 KD 재현성 점검

구버전 코드는 student soft-assignment logits까지 `no_grad` 안에서 계산해 KD
loss가 student에 gradient를 전달하지 않았다. 따라서 구버전 문서의 “KD 적용”
결과는 실제로는 triplet 학습 효과로 보는 것이 타당하다.

현재 코드는 student logits를 gradient graph 안에서 계산하도록 고쳤지만,
teacher와 student가 독립 K-means로 만든 cluster index는 의미가 정렬되어 있지
않다. 정렬하지 않은 cluster-logit KD는 구버전과 동등한 비교도, 의미가 보장된
KD도 아니다. 이 때문에 공식 재현 run은 `--kd-weight 0`으로 실행했다.

### 6.4 ImageNet V1 + 15% pruning 재현

run: `artifacts/runs/20260831_185123_mobilenetv2_netvlad`

공식 선택 기준인 val R@5 best는 epoch 2다.

| Split | 모델 | R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|
| val | zero-pruned | 0.7169 | 0.8864 | 0.9298 | 0.9566 |
| val | reduced | 0.7137 | 0.8838 | 0.9294 | 0.9572 |
| test | zero-pruned | 0.7073 | 0.8484 | 0.8928 | 0.9256 |
| test | reduced | 0.7039 | 0.8474 | 0.8922 | 0.9241 |

실제 구조 축소에 따른 test R@1 손실은 약 0.34%p였다. 이 run은 epoch 6까지
수행했으며, reduced 학습 가능 파라미터 462,560개로 압축률은 78.3088%였다.

## 7. 26% pruning, 30 epoch 실험

run: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26`

### 7.1 전체 validation 학습 곡선

| Epoch | Loss | LR | R@1 | R@5 | R@10 | R@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2552 | 1e-4 | 0.6600 | 0.8456 | 0.8967 | 0.9381 |
| 2 | 0.2318 | 1e-4 | 0.7115 | **0.8788** | **0.9211** | **0.9482** |
| 3 | 0.2121 | 1e-4 | 0.7099 | 0.8774 | 0.9177 | 0.9469 |
| 4 | 0.1950 | 1e-4 | **0.7191** | 0.8742 | 0.9168 | 0.9457 |
| 5 | 0.1754 | 1e-4 | 0.6906 | 0.8555 | 0.9018 | 0.9363 |
| 6 | 0.1637 | 5e-5 | 0.6939 | 0.8595 | 0.9055 | 0.9384 |
| 7 | 0.1581 | 5e-5 | 0.6940 | 0.8558 | 0.9035 | 0.9341 |
| 8 | 0.1505 | 5e-5 | 0.6869 | 0.8513 | 0.9005 | 0.9345 |
| 9 | 0.1442 | 5e-5 | 0.6749 | 0.8496 | 0.8951 | 0.9310 |
| 10 | 0.1385 | 5e-5 | 0.6941 | 0.8569 | 0.9056 | 0.9391 |
| 11 | 0.1361 | 2.5e-5 | 0.7041 | 0.8661 | 0.9100 | 0.9399 |
| 12 | 0.1346 | 2.5e-5 | 0.6956 | 0.8552 | 0.8992 | 0.9327 |
| 13 | 0.1299 | 2.5e-5 | 0.6845 | 0.8478 | 0.8959 | 0.9353 |
| 14 | 0.1300 | 2.5e-5 | 0.6805 | 0.8441 | 0.8951 | 0.9299 |
| 15 | 0.1259 | 2.5e-5 | 0.6814 | 0.8495 | 0.8943 | 0.9309 |
| 16 | 0.1239 | 1.25e-5 | 0.6719 | 0.8442 | 0.8948 | 0.9295 |
| 17 | 0.1247 | 1.25e-5 | 0.6801 | 0.8499 | 0.9002 | 0.9365 |
| 18 | 0.1220 | 1.25e-5 | 0.6769 | 0.8488 | 0.8988 | 0.9336 |
| 19 | 0.1237 | 1.25e-5 | 0.6798 | 0.8498 | 0.9014 | 0.9356 |
| 20 | 0.1207 | 1.25e-5 | 0.6807 | 0.8511 | 0.9025 | 0.9366 |
| 21 | 0.1205 | 6.25e-6 | 0.6795 | 0.8506 | 0.9000 | 0.9343 |
| 22 | 0.1182 | 6.25e-6 | 0.6877 | 0.8594 | 0.9059 | 0.9397 |
| 23 | 0.1212 | 6.25e-6 | 0.6853 | 0.8555 | 0.9025 | 0.9364 |
| 24 | 0.1197 | 6.25e-6 | 0.6890 | 0.8546 | 0.9027 | 0.9394 |
| 25 | 0.1185 | 6.25e-6 | 0.6743 | 0.8442 | 0.8958 | 0.9340 |
| 26 | 0.1175 | 3.125e-6 | 0.6857 | 0.8540 | 0.9047 | 0.9405 |
| 27 | 0.1166 | 3.125e-6 | 0.6847 | 0.8558 | 0.9094 | 0.9384 |
| 28 | 0.1186 | 3.125e-6 | 0.6839 | 0.8515 | 0.8984 | 0.9366 |
| 29 | 0.1187 | 3.125e-6 | 0.6744 | 0.8488 | 0.8988 | 0.9361 |
| 30 | 0.1192 | 3.125e-6 | 0.6760 | 0.8478 | 0.8992 | 0.9353 |

R@10과 R@20도 epoch 2가 최고였다. loss 최저값은 epoch 27의 0.1166이지만
recall 최고점과 일치하지 않는다.

### 7.2 구조 축소 전후 Recall

| Checkpoint | Split | 상태 | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---:|---:|---:|---:|
| R@5 best, epoch 2 | val | zero-pruned | 0.7115 | 0.8788 | 0.9211 | 0.9482 |
| R@5 best, epoch 2 | val | reduced | 0.7031 | 0.8755 | 0.9194 | 0.9461 |
| R@5 best, epoch 2 | test | zero-pruned | 0.6931 | 0.8461 | 0.8906 | 0.9244 |
| R@5 best, epoch 2 | test | reduced | 0.6891 | 0.8417 | 0.8879 | 0.9222 |
| R@1 best, epoch 4 | val | zero-pruned | 0.7191 | 0.8742 | 0.9168 | 0.9457 |
| R@1 best, epoch 4 | val | reduced | 0.7131 | 0.8700 | 0.9139 | 0.9435 |
| R@1 best, epoch 4 | test | zero-pruned | 0.6979 | 0.8445 | 0.8928 | 0.9274 |
| R@1 best, epoch 4 | test | reduced | 0.6920 | 0.8392 | 0.8906 | 0.9250 |
| epoch 30 | val | zero-pruned | 0.6760 | 0.8478 | 0.8992 | 0.9353 |
| epoch 30 | val | reduced | 0.6678 | 0.8446 | 0.8950 | 0.9322 |
| epoch 30 | test | zero-pruned | 0.6463 | 0.8235 | 0.8722 | 0.9102 |
| epoch 30 | test | reduced | 0.6401 | 0.8201 | 0.8693 | 0.9089 |

구조 축소에 따른 test 변화는 작다.

| Checkpoint | ΔR@1 | ΔR@5 | ΔR@10 | ΔR@20 |
|---|---:|---:|---:|---:|
| epoch 2 | -0.40%p | -0.44%p | -0.26%p | -0.22%p |
| epoch 4 | -0.59%p | -0.53%p | -0.22%p | -0.23%p |
| epoch 30 | -0.62%p | -0.34%p | -0.29%p | -0.13%p |

### 7.3 파라미터, descriptor, 파일 크기

| 모델 | Encoder ch | Descriptor dim | Total params | Trainable params | Zero params |
|---|---:|---:|---:|---:|---:|
| ResNet18 기준 | 256 | 16,384 | 2,815,552 | 2,132,480 | 0 |
| MobileNetV2 zero-pruned | 320 | 20,480 | 1,852,672 | 514,880 | 79,846 |
| MobileNetV2 structurally reduced | 237 | 15,168 | 1,762,202 | 424,410 | 0 |

| 비교 | Total 감소율 | Trainable 감소율 |
|---|---:|---:|
| reduced vs ResNet18 | 37.4118% | **80.0978%** |
| reduced vs MobileNetV2 | 4.8832% | 17.5711% |

descriptor 차원은 20,480에서 15,168로 25.9375% 줄었다. reduced checkpoint는
재개할 수 없는 원본 optimizer/scheduler 상태를 제외한 배포용 payload로 저장했다.

| 파일 | Bytes | 원본 학습 checkpoint 대비 감소 |
|---|---:|---:|
| `best.pt` | 9,688,882 | - |
| `best_reduced.pt` | 7,282,617 | 24.84% |
| `best_r1.pt` | 9,705,985 | - |
| `best_r1_reduced.pt` | 7,283,559 | 24.96% |
| `latest.pt` | 9,701,308 | - |
| `latest_reduced.pt` | 7,283,245 | 24.93% |

파일 크기 감소에는 구조 축소와 학습 전용 상태 제거가 모두 포함되므로 공식
파라미터 압축률로 사용하지 않는다.

### 7.4 15%와 26% 비교

| 설정/선택 기준 | 채널 | Trainable | ResNet 대비 압축 | Test R@1 | Test R@5 |
|---|---:|---:|---:|---:|---:|
| 15%, R@5 best reduced | 272 | 462,560 | 78.3088% | 0.7039 | 0.8474 |
| 26%, R@5 best reduced | 237 | 424,410 | **80.0978%** | 0.6891 | 0.8417 |
| 26%, R@1 best reduced | 237 | 424,410 | **80.0978%** | 0.6920 | 0.8392 |

동일한 R@5 선택 기준으로 26%는 15%보다 test R@1이 1.48%p, R@5가
0.57%p 낮다. 즉 목표 압축률을 넘기는 대가로 소폭의 Recall 손실이 발생했다.

## 8. 재현 명령

공용 서버 기본 규칙은 물리 GPU 2지만, 이번 run을 동일하게 재현하려면 명시적으로
GPU 0을 지정한다.

```bash
source .venv/bin/activate
./scripts/check_resources.sh

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

중단 시 같은 전체 epoch 수를 유지하고 `latest.pt`로 재개한다.

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
  --checkpoint artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/latest.pt
```

구조 축소와 평가는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=0 robot-compression reduce \
  --checkpoint artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/best_r1.pt \
  --output artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/best_r1_reduced.pt \
  --pruning-ratio 0.26

CUDA_VISIBLE_DEVICES=0 robot-compression test \
  --checkpoint artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/best_r1_reduced.pt \
  --split test \
  --cache-batch-size 48 \
  --workers 8
```

## 9. 산출물

- 실행 조건: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/run_metadata.json`
- epoch별 원시 지표: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/metrics.csv`
- 축소 전후 원시 평가: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/evaluation.json`
- R@5 best 배포 모델: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/best_reduced.pt`
- R@1 best 배포 모델: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/best_r1_reduced.pt`
- epoch 30 배포 모델: `artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26/checkpoints/latest_reduced.pt`

## 10. 해석 시 한계와 다음 실험

- 80% 목표는 trainable parameter 기준이다. 전체 파라미터 기준 목표라면 현재
  구조는 달성하지 못했다.
- 26% 설정은 30 epoch 1회, seed 42 결과다. 신뢰구간을 얻으려면 여러 seed가
  필요하다.
- 현재 pruning은 마지막 descriptor channel만 대상으로 한다. 전체 파라미터
  압축률을 크게 높이려면 MobileNetV2 중간 inverted residual channel이나
  NetVLAD cluster 수까지 포함한 의존성-aware pruning이 필요하다.
- teacher/student NetVLAD KD를 다시 사용하려면 cluster matching 또는 embedding
  차원의 KD처럼 의미가 정렬된 objective를 먼저 구현해야 한다.
- 파라미터 수와 checkpoint 크기만 측정했다. 실제 로봇 배포 판단에는 ONNX/TensorRT
  변환 후 지연시간, peak memory, 전력 측정이 추가로 필요하다.
- 현재 직접 재현한 26% reduced 결과는 구버전 reduced test R@1 0.7199보다
  2.79~3.08%p 낮다. 압축률 목표는 달성했지만 구버전의 절대 Recall까지 완전히
  재현한 것은 아니다.
