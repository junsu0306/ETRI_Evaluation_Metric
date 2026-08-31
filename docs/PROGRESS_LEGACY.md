# 로봇 압축 진행 상황 정리 (NetVLAD 기반 최적화율 실험)

## 1. 성능 지표 정의 및 목표치

- **지표 정의**: 로봇 단위작업 지능처리 신경망에 최적구조 학습 기술을 적용한 후 모델 크기를 A, 본래 모델 크기를 B라 할 때 → **100 × (B − A) / B (%)**
- **기존 기술 수준**: IROS에서 Visual Place Recognition용 ResNet18을 압축, 학습 가능 파라미터 수 기준 **71.18%** 최적화율 (Pittsburgh-30k, 싱가포르 Nanyang 공대)
- **본 과제 목표**: 이기종 온디바이스 실행환경을 고려한 구조 학습 기반 모델 생성·실행 기법으로 **80%** 달성

---

## 2. 실험 환경

- 코드베이스: pytorch-NetVlad (https://github.com/Nanne/pytorch-NetVlad/)
- 데이터셋: Pittsburgh, pooling = NetVLAD, num_clusters = 64
- 학습 명령어

  ```
  python main_variant.py --mode=train --arch=resnet-18 --pooling=netvlad --num_clusters=64 --pruning
  ```

- **비고**: 학습 전 반드시 클러스터링을 선행해야 함 (`--mode=cluster`). 원 논문에도 해당 절차가 기술되어 있을 것으로 보임.

### Run 이름 대응표

| Run | 내용 |
|---|---|
| May06_07-30-02_resnet-18_netvlad | ResNet-18, 프루닝 없음 (baseline) |
| May01_14-48-56_resnet-18_netvlad | ResNet-18, 마지막 블록만 프루닝 |
| May20_13-57-55_mobilenetv2_netvlad | MobileNetV2 (원본, 30 epoch) |
| Aug21_15-48-05_mobilenetv2_netvlad | MobileNetV2 + 최적구조 학습(pruning → reducing) |

---

## 3. 인식 성능 (Recall@N)

### ResNet-18 계열 — best checkpoint 기준

| 모델 | Split | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| ResNet-18 (프루닝 없음) | val | 0.7805 | 0.9172 | 0.9451 | 0.9670 |
| ResNet-18 (프루닝 없음) | test | 0.7399 | 0.8750 | 0.9117 | 0.9420 |
| ResNet-18 (마지막 블록 프루닝) | val | 0.6789 | 0.8414 | 0.8825 | 0.9203 |
| ResNet-18 (마지막 블록 프루닝) | test | 0.6504 | 0.8047 | 0.8555 | 0.9004 |

→ ResNet-18을 직접 프루닝하면 test R@1이 0.7399 → 0.6504로 크게 떨어져, ResNet만으로 목표 최적화율을 맞추기 어려움을 확인.

### MobileNetV2 (Distillation 적용)

| 모델 | Split | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| MobileNetV2 원본 (30 epoch) | val | 0.7128 | 0.8829 | 0.9264 | 0.9572 |
| MobileNetV2 원본 (30 epoch) | test | 0.7061 | 0.8521 | 0.8994 | 0.9330 |
| MobileNetV2 + pruning | test | 0.7193 | 0.8688 | 0.9096 | 0.9398 |
| MobileNetV2 + reducing | test | 0.7199 | 0.8662 | 0.9080 | 0.9384 |
| MobileNetV2, reducing 이전 | val | 0.7340 | 0.8872 | 0.9246 | 0.9528 |
| MobileNetV2, reducing 이후 | val | 0.7337 | 0.8933 | 0.9292 | 0.9579 |

→ pruning 후 reducing(실제 구조 축소)을 적용해도 Recall 변화가 거의 없음(val R@1 0.7340 → 0.7337). 즉 **성능 손실 없이 파라미터만 실제로 줄어드는 것을 확인**.

---

## 4. 파라미터 수 및 최적화율

| 모델 | Total params | Trainable params | Zero params |
|---|---|---|---|
| ResNet-18 (기준 모델, B) | 2,815,552 | **2,132,480** | 0 |
| MobileNetV2 (distillation 직후) | 1,852,672 | 514,880 | 0 |
| MobileNetV2 (reducing 이후, A) | 1,800,352 | **462,560** | 0 |

### 최적화율 계산 (학습 가능 파라미터 기준)

- (2,132,480 − 462,560) / 2,132,480 × 100 = **78.3%**
- 목표 80% 달성 기준선: 학습 가능 파라미터 **426,496** 이하
- 참고: distillation만 적용한 시점(514,880)의 최적화율은 약 75.9% 수준

---

## 5. 진행 상황 요약 (7월 성과교류회 이후)

- 성능목표치 검증은 **Pittsburgh + ResNet18** 조합을 기준으로 진행
- ResNet18 자체 프루닝만으로는 목표 최적화율 도달이 어려워, **작은 모델로 Distillation한 뒤 최적구조 학습을 적용**하는 방향으로 전환
- ResNet18 → MobileNetV2 / MobileNetV3로 각각 distillation 수행
- **MobileNetV2**: Distillation만으로도 최적화율이 거의 80%에 근접, 이후 최적구조 학습(pruning + reducing)까지 적용해 **78.3%** 확보
- **MobileNetV3**: Distillation 후 최적구조 학습 진행 중

**구조 관련 관찰**: 백본에서 프루닝을 적용한다면 마지막 레이어(MobileNetV3 기준 마지막 InvertedResidual 블록: 576ch depthwise conv + SqueezeExcitation)를 대상으로 하는 것이 의미 있음.

---

## 6. 남은 과제

- 목표 80% 달성까지 학습 가능 파라미터 기준 약 36,000개(462,560 → 426,496) 추가 축소 필요
- MobileNetV3 기반 최적구조 학습 결과 확보 후 V2와 비교
- reducing 이후 모델에 대한 test split 재측정 및 Recall 유지 여부 최종 확인
