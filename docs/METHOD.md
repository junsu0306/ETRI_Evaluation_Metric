# 실험 방법

## 데이터와 평가

Pittsburgh30k의 고정 train/validation/test split을 사용한다.
DB/query 수는 train 10,000/7,416, validation 10,000/7,608, test 10,000/6,816이다.
학습 positive 후보는 10m 이내, negative 후보는 25m 밖에서 선택하며,
query당 positive 1개와 hard negative 최대 10개를 사용한다.
평가는 각 split의 전체 DB를 검색하고, 상위 N개 중 25m 이내 장소가 있으면 성공으로 계산한다.

## Teacher와 student

- Teacher: ImageNet 초기화 ResNet18의 layer3 출력(256채널) + NetVLAD64, descriptor 16,384차원.
- Student: ImageNet V1 MobileNetV3-Large에서 마지막 160→960 convolution과 classifier 제외,
  출력 160채널 + NetVLAD64, descriptor 10,240차원.
- Student의 마지막 InvertedResidual과 NetVLAD만 학습한다. 앞부분 parameter와 BatchNorm 통계는 고정한다.
- Teacher는 seed 42로 19 epoch 완료 후 수동 종료했으며 validation R@1 최고인 epoch 14를 선택했다.
  Teacher의 Val R@1은 87.8155%, Test R@1은 85.5927%다.
- Teacher SHA-256: `7340574ae75d08e07fd324e82d2932842205685c8edcd4a8719cc16309f3f897`.

## Pruning과 실제 구조 축소

마지막 residual 블록의 expansion 960채널 중 L2 norm이 작은 채널을 선택한다.
Expansion convolution 출력·BN, depthwise convolution·BN, SE의 fc1 입력·fc2 출력,
projection convolution 입력에 같은 채널 index를 적용한다. 매 optimizer step 후 zeroing을 다시 적용한다.
블록의 residual 출력 160채널과 NetVLAD descriptor 크기는 유지한다.

| 목표 trainable 압축률 | 제거/유지 채널 | Reduced total | Reduced trainable | 실제 trainable 감소율 | Total 감소율 |
|---|---|---:|---:|---:|---:|
| 72% | 266 / 694 | 2,616,132 | 597,060 | 72.0016% | 7.0828% |
| 76% | 369 / 591 | 2,530,642 | 511,570 | 76.0106% | 10.1192% |
| 80% | 472 / 488 | 2,445,152 | 426,080 | 80.0195% | 13.1555% |

기준 teacher는 total 2,815,552개, trainable 2,132,480개다.
공식 압축률은 `100 × (1 − student trainable / teacher trainable)`로 계산한다.
CLI의 `--pruning-ratio`는 **expansion 채널 제거 비율**이며 위의 공식 압축률과 다르다.
구조 축소에서는 이미 zeroing된 채널만 제거하도록 `--pruning-ratio 0`을 사용한다.

## 관계 KD와 대조군

Teacher/student의 최종 descriptor로 같은 query와 후보 사이의 cosine similarity를 계산한다.
온도 T=0.1의 softmax 분포에 대해 `KL(teacher || student)`를 최소화하며 T²를 곱하지 않는다.
서로 다른 descriptor 차원이나 독립적으로 학습된 cluster index를 직접 정렬하지 않는다.

```text
L_triplet = mean(max(||q − p||₂ − ||q − n||₂ + sqrt(0.1), 0))
L_KD = KL(softmax(cosine_teacher / 0.1) || softmax(cosine_student / 0.1))
KD-on:  L = 0.5 × L_triplet + 0.5 × L_KD
KD-off: L = 0.5 × L_triplet
```

Teacher는 gradient를 계산하지 않는다. KD-on은 고정 teacher의 train descriptor cache를 재사용한다.
KD-off도 초기 난수 소비를 맞추기 위해 teacher를 생성하지만 CPU로 옮기고 추론·cache·KD 계산을 생략한다.
공개 `kd_control` 모듈은 실제 실험의 process-local criterion wrapper를 분리한 것이다.
일반 CLI의 `--kd-weight 0`만 사용하면 triplet 계수가 1이 되므로 최종 대조군과 다르다.

## 학습과 선택

| 설정 | 값 |
|---|---|
| Seed / query batch / cache batch / workers | 42 / 1 / 48 / 8 |
| Optimizer | SGD, lr 1e-4, momentum 0.9, weight decay 1e-3 |
| Scheduler | StepLR(step_size=5, gamma=0.5) |
| Epoch 상한 / early stopping | 50 / validation R@1 patience 10 |
| 선택 | validation R@1 최고 `best_r1.pt`; 동점은 미개선 |
| NetVLAD | backbone별 descriptor 50,000개로 64-cluster centroid 초기화 |

72/76/80% KD-on은 각각 50/34/42 epoch를 완료하고 46/24/32 epoch를 선택했다.
80% KD-off는 37 epoch 완료, 27 epoch 선택이다. Test는 선택·종료에 사용하지 않았다.
측정은 RTX A5000에서 수행했으며 GPU 번호는 다른 환경의 재현 조건이 아니다.
모델별 설정은 [공개 결과 출처](../results/provenance.json)와 `results/configs/`에 있다.
