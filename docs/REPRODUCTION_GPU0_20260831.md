# Pittsburgh30k pruning/reduction 재현 결과 (GPU 0)

## 실행 조건

- 실행일: 2026-08-31
- 물리 GPU: NVIDIA RTX A5000 0번 (`CUDA_VISIBLE_DEVICES=0`, 논리 `cuda:0`)
- 데이터: Pittsburgh30k train/val/test
- NetVLAD cluster: 64
- seed: 42
- optimizer: SGD, learning rate `1e-4`, momentum `0.9`, weight decay `1e-3`
- pruning: MobileNetV2 마지막 projection output 320채널 중 15%(48채널)
- MobileNetV2 pretrained weights: `IMAGENET1K_V1`
- MobileNetV2 centroid: 50,000 descriptors로 새로 생성

이번 실행은 사용자의 명시적 요청에 따라 저장소 기본값인 물리 GPU 2번 대신
GPU 0번을 사용했다. 시작 전 GPU 0의 가용 메모리는 약 22.6 GiB였다.

## 재현 중 확인한 차이

1. 구버전 torchvision의 `pretrained=True`는 MobileNetV2
   `IMAGENET1K_V1`을 사용하지만 현재 torchvision의 `DEFAULT`는
   `IMAGENET1K_V2`다. V2로 학습했을 때 epoch 2 val R@1이 약 0.38에
   머물렀고, V1과 V1 전용 centroid를 사용하자 epoch 1부터 0.6677로
   회복됐다.
2. 구버전 KD 코드는 student soft-assignment logits까지 `no_grad` 안에서
   계산해 KD gradient가 student에 전달되지 않았다. 또한 teacher와 student가
   독립적으로 생성한 K-means cluster 번호는 의미가 정렬돼 있지 않으므로,
   현재의 cluster-logit KD는 구버전 결과와 동등한 학습이 아니다.
3. 따라서 구버전 성능을 실질적으로 재현하는 비교 run은
   MobileNetV2 `triplet loss + pruning`으로 실행했다.

중단한 비교 run도 삭제하지 않고 `artifacts/runs`에 보존했다.

- 실제 soft-assignment KD + V2: `20260831_174635_mobilenetv2_netvlad`
- triplet + V2: `20260831_182807_mobilenetv2_netvlad`
- 최종 triplet + V1: `20260831_185123_mobilenetv2_netvlad`

## Teacher 결과

ResNet18 teacher는 epoch 4에서 기존 기록 수준에 도달해 그 시점의
`best.pt`를 사용했다.

| Split | R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|
| val, 이번 실행 | 0.7896 | 0.9114 | 0.9426 | 0.9661 |
| val, 구버전 기록 | 0.7805 | 0.9172 | 0.9451 | 0.9670 |
| test, 이번 실행 | 0.8078 | 0.9024 | 0.9300 | 0.9520 |
| test, 구버전 기록 | 0.7399 | 0.8750 | 0.9117 | 0.9420 |

Checkpoint:
`artifacts/runs/20260831_164217_resnet18_netvlad/checkpoints/best.pt`

## MobileNetV2 공식 best(R@5) reduction 전후

공식 checkpoint 선택 방식은 구버전과 동일하게 val R@5 기준이며, 이번
실행의 best는 epoch 2다.

| Split | 모델 | R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|
| val | zero-pruned | 0.7169 | 0.8864 | 0.9298 | 0.9566 |
| val | structurally reduced | 0.7137 | 0.8838 | 0.9294 | 0.9572 |
| test | zero-pruned | 0.7073 | 0.8484 | 0.8928 | 0.9256 |
| test | structurally reduced | 0.7039 | 0.8474 | 0.8922 | 0.9241 |

Reduction에 따른 변화는 다음과 같다.

| Split | ΔR@1 | ΔR@5 | ΔR@10 | ΔR@20 |
|---|---:|---:|---:|---:|
| val | -0.0032 | -0.0026 | -0.0004 | +0.0005 |
| test | -0.0034 | -0.0010 | -0.0006 | -0.0015 |

따라서 실제 구조 축소 자체의 Recall 손실은 R@1 기준 약 0.3%p로 작았다.

## R@1이 가장 높았던 중간 checkpoint

epoch 4는 공식 R@5 best는 아니지만 R@1이 더 높아 별도로 보존했다.

| Split | 모델 | R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|
| val | zero-pruned | 0.7253 | 0.8824 | 0.9228 | 0.9490 |
| val | structurally reduced | 0.7224 | 0.8785 | 0.9217 | 0.9490 |
| test | zero-pruned | 0.7042 | 0.8549 | 0.9013 | 0.9297 |
| test | structurally reduced | 0.7014 | 0.8539 | 0.8991 | 0.9287 |

## 구버전 reduced 결과와 비교

| Split | 결과 | R@1 | R@5 | R@10 | R@20 |
|---|---|---:|---:|---:|---:|
| val | 이번 실행, 공식 best reduced | 0.7137 | 0.8838 | 0.9294 | 0.9572 |
| val | 구버전 reduced | 0.7337 | 0.8933 | 0.9292 | 0.9579 |
| test | 이번 실행, 공식 best reduced | 0.7039 | 0.8474 | 0.8922 | 0.9241 |
| test | 구버전 reduced | 0.7199 | 0.8662 | 0.9080 | 0.9384 |

이번 실행은 reduction 전후 성능 유지 경향은 재현했지만 절대 Recall은
완전히 재현하지 못했다. 공식 best reduced 기준 차이는 val R@1 -2.00%p,
test R@1 -1.60%p다.

## 파라미터와 checkpoint 크기

| 모델 | Total params | Trainable params | Checkpoint bytes |
|---|---:|---:|---:|
| zero-pruned MobileNetV2 | 1,852,672 | 514,880 | 9,688,498 |
| structurally reduced MobileNetV2 | 1,800,352 | 462,560 | 9,498,650 |

- MobileNetV2 자체 대비 total parameter 감소: 약 2.82%
- MobileNetV2 자체 대비 trainable parameter 감소: 약 10.16%
- ResNet18 teacher 대비 trainable parameter 감소: 약 78.31%
- reduced 모델의 파라미터 tensor에는 pruning으로 생긴 zero가 남아 있지 않다.

Checkpoint 파일 크기 감소가 파라미터 감소보다 작은 이유는 payload에 optimizer
state와 실험 메타데이터도 포함되기 때문이다.

## 산출물

- 공식 best 원본: `artifacts/runs/20260831_185123_mobilenetv2_netvlad/checkpoints/best.pt`
- 공식 best reduced: `artifacts/runs/20260831_185123_mobilenetv2_netvlad/checkpoints/best_reduced.pt`
- epoch 4 원본: `artifacts/runs/20260831_185123_mobilenetv2_netvlad/checkpoints/epoch_004.pt`
- epoch 4 reduced: `artifacts/runs/20260831_185123_mobilenetv2_netvlad/checkpoints/epoch_004_reduced.pt`
- V1 centroid: `artifacts/centroids/mobilenetv2_imagenet1k_v1_pitts30k_64_desc_cen.hdf5`
