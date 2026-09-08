# 최종 MobileNetV3-Large 관계 기반 KD 실험 계획

작성일: 2026-09-07

> 2026-09-08 상태 안내: 이 V3-Large 계획은 보류 상태다. 별도로 수행한
> MobileNetV2 15% pruning KD-on/off 학습은 사용자 요청으로 모두 종료했다.
> [현재 결과 보고서](KD_ABLATION_FINAL_20260908.md)를 참고한다.
> 아래 additive KD 식은 이 계획의 식이며, 실제 V2 실행의 convex-combination 식과 구분한다.

## 1. 최종 결과 범위

이 프로젝트의 최종 결과 모델은 **MobileNetV3-Large + NetVLAD**로 확정한다.
ResNet18 + NetVLAD를 고정 teacher이자 압축률 기준 모델로 사용하고, 관계 기반
knowledge distillation(KD)과 dependency-aware structural channel reduction을
적용한다.

최종 비교점은 ResNet18의 학습 가능 파라미터 2,132,480개 대비 압축률
72%, 76%, 80% 세 조건이다. 세 조건 모두 같은 백본, centroid, teacher, seed,
optimizer와 checkpoint 선택 규칙을 사용한다.

## 2. 모델과 pruning 단위

MobileNetV3-Large의 torchvision `features`에서 마지막 `160→960` convolution과
classifier를 제외한다. encoder 출력은 마지막 InvertedResidual의 160채널이며,
NetVLAD cluster 수는 64, global descriptor 크기는 10,240이다.

마지막 InvertedResidual은 `160→160` identity shortcut을 사용하므로 출력 채널을
직접 제거하지 않는다. 블록 내부의 960개 expansion channel을 pruning하고 다음
의존 축을 같은 index로 함께 줄인다.

1. expansion convolution 출력과 BatchNorm
2. 5×5 depthwise convolution과 BatchNorm
3. Squeeze-Excitation `fc1` 입력과 `fc2` 출력
4. projection convolution 입력

이 방식은 블록 출력 160채널과 residual add, 이후 NetVLAD 입력을 그대로 유지한다.

| 목표 | 제거/전체 expansion ch | 유지 ch | Trainable params | Trainable 압축률 | Total params | Total 압축률 |
|---|---:|---:|---:|---:|---:|---:|
| 72% | 266/960 | 694 | 597,060 | 72.0016% | 2,616,132 | 7.0828% |
| 76% | 369/960 | 591 | 511,570 | 76.0106% | 2,530,642 | 10.1192% |
| 80% | 472/960 | 488 | 426,080 | 80.0195% | 2,445,152 | 13.1555% |

압축률은 기존 과제 정의와 동일하게 trainable parameter 기준으로 판정한다. 모델의
전체 파라미터 수와 checkpoint 파일 크기도 함께 기록하되 공식 압축률과 구분한다.

## 3. 관계 기반 KD

teacher와 student의 독립적인 NetVLAD cluster logit은 직접 비교하지 않는다.
각 학습 query에 대해 positive와 hard negatives의 cosine similarity를 계산하고,
teacher와 student의 검색 순서 확률분포를 KL divergence로 맞춘다.

```text
teacher scores = cosine(q_t, [p_t, n1_t, ...]) / T
student scores = cosine(q_s, [p_s, n1_s, ...]) / T
L_KD = KL(softmax(teacher scores) || softmax(student scores))
L_total = L_triplet + lambda_KD * L_KD
```

- `temperature T`: 0.1
- `lambda_KD`: 0.5
- teacher: eval mode, gradient 비활성화
- student triplet loss: 기존 크기를 유지하고 KD를 보조항으로 더함
- teacher descriptor는 고정 cache를 한 번 생성해 세 실험에서 재사용

학습 전용 `1×1 Conv(160→256)`로 ResNet18과 MobileNetV3-Large의 local feature
차원을 맞추는 projection KD도 가능하다. 그러나 두 encoder의 공간 해상도
`14×14`와 `7×7`까지 정렬해야 하며 KD 방식 자체가 추가 실험 변수가 된다.
최종 세 조건은 추가 head와 공간 정렬 없이 검색 관계를 직접 보존하는 위 관계
KD만 사용하고, projection KD는 별도 ablation 범위로 남긴다.

## 4. 실행 단계

### 단계 A. ResNet18 seed 재검증

기존 seed 42와 같은 4 epoch 조건으로 seed 123 ResNet18을 다시 학습한다.
두 모델 중 validation R@5가 높은 checkpoint를 teacher로 선택한다. test 결과는
teacher 선택에 사용하지 않고 선택 완료 후 보고용으로 한 번 평가한다.

### 단계 B. MobileNetV3-Large centroid

`IMAGENET1K_V1` pretrained MobileNetV3-Large encoder와 Pittsburgh30k train DB를
사용해 descriptor 50,000개를 표본화하고 64-cluster K-means centroid를 생성한다.

### 단계 C. 세 압축 조건 학습

각 조건은 다음 공통 설정으로 30 epoch 수행한다.

| 항목 | 값 |
|---|---:|
| Dataset | Pittsburgh30k train/val/test |
| Seed | 42 |
| Epoch | 30 |
| Batch size | 1 query group |
| Cache batch size | 48 |
| Workers | 8 |
| Optimizer | SGD |
| Learning rate | 1e-4 |
| Momentum | 0.9 |
| Weight decay | 1e-3 |
| Scheduler | StepLR(5, 0.5) |
| Triplet margin | sqrt(0.1) |
| KD weight / temperature | 0.5 / 0.1 |

ImageNet pretrained 앞단은 parameter뿐 아니라 BatchNorm running statistics도
고정한다. 각 optimizer step 후 목표 expansion channel set을 다시 zeroing한다.

### 단계 D. 구조 축소 및 평가

매 epoch validation R@1/R@5/R@10/R@20을 기록한다. 기존 프로젝트의 공식 규칙에
따라 validation R@5 best를 최종 checkpoint로 선택하고, R@1 best도 보조 결과로
보존한다. 두 checkpoint를 실제 구조로 줄인 뒤 validation과 test 전체를 다시
평가한다.

최종 표에는 다음을 포함한다.

- 목표/실제 trainable parameter 압축률
- 유지 expansion channel 수
- total/trainable parameter 수
- validation 및 test Recall@1/5/10/20
- zero-pruned 대비 structurally reduced 성능 변화
- epoch별 triplet/KD/total loss

## 5. 재현성과 판정 규칙

- 모든 GPU 실행은 시작 직전 공용 서버 RAM, 디스크, GPU 가용량을 확인한다.
- 기본 물리 GPU 2를 우선 사용하고, 사용자 지정 이력이 있는 물리 GPU 0을
  병렬 실험에 사용한다. 각 run 시작 직전에 해당 GPU의 여유 메모리를 다시
  확인한다.
- test split을 hyperparameter 또는 checkpoint 선택에 사용하지 않는다.
- 세 압축 조건 사이에서는 압축률 이외의 조건을 바꾸지 않는다.
- dataset, centroid, teacher checkpoint와 실행 인자를 run metadata에 기록한다.
- 최종 결과는 zero masking checkpoint가 아니라 실제 구조가 축소된 checkpoint로
  판정한다.

## 6. 실행 기록

2026-09-07에 다음 실행을 시작했다. 이 절은 학습 완료 후 최종 checkpoint와
평가 결과로 갱신한다.

| 항목 | 경로 또는 상태 |
|---|---|
| ResNet18 seed 123 재검증 | `artifacts/runs/20260907_145736_resnet18_netvlad` |
| MobileNetV3-Large centroid | `artifacts/centroids/mobilenetv3large_pitts30k_64_desc_cen.hdf5` |
| 72% 사전 실행 | `20260907_160951...`: 사용자 요청으로 epoch 1 완료 전 중단, 결과 제외 |
| 76% 사전 실행 | `20260907_161009...`: 사용자 요청으로 epoch 1 완료 전 중단, 결과 제외 |
| 80% 사전 실행 | `20260907_161001...`: 사용자 요청으로 epoch 1 완료 전 중단, 결과 제외 |
| 72%·76%·80% 최종 run | 일시 중지, 재시작 시 새 run으로 처음부터 수행 |

centroid SHA-256은
`9346d3716b0f14160663c042ee98e136f48de1285c274191eee2c3a244fee565`다.
seed 123 ResNet18은 epoch 4에서 validation R@5 0.912461을 기록해 seed 42의
0.911409를 0.001052만큼 갱신했다. 이에 따라 seed 123 checkpoint를 최종
teacher로 확정했으며 SHA-256은
`bbab64a55eede03cc27ccd89cd764146adf8a4f515077f5f40b0292a276ee06c`다.

seed 42를 임시 teacher로 사용했던 `20260907_153030...` 및
`20260907_153715...` run은 teacher 확정 즉시 중단했으며 최종 결과에서 제외한다.
확정된 seed 123 teacher로 시작한 세 사전 run도 사용자 요청에 따라 epoch 1 완료
전에 중단했다. 저장된 epoch checkpoint와 공식 결과는 없으며, 다음 실행에서는
동일한 계획으로 새 run을 처음부터 시작한다.

| ResNet18 seed | 선택 epoch | Val R@1 | Val R@5 | Test R@1 | Test R@5 |
|---:|---:|---:|---:|---:|---:|
| 42 | 4 | 0.78956 | 0.91141 | 0.8078 | 0.9024 |
| **123** | **4** | **0.78996** | **0.91246** | **0.80692** | **0.90185** |

두 seed의 test 결과도 거의 같으므로 기존 ResNet18 기준선의 재현성을 확인했다.
교사 선택에는 test가 아니라 validation R@5만 사용했다.
