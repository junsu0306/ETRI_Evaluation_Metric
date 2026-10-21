# 최종 실험 결과

2026-09-11 완료된 측정값이다. 공개 패키지 정리 과정에서 학습이나 정확도 평가를 다시 수행하지 않았다.

## 실제 구조 축소 후 test

| 모델 | 선택 epoch | Test R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| 공식 ResNet18 teacher | 14 | 85.5927% | 91.9748% | 93.9554% | 95.3492% |
| V3 72% + KD | 46 | 81.8369% | 90.3756% | 92.8404% | 95.0411% |
| V3 76% + KD | 24 | 81.7488% | 90.3462% | 92.7377% | 94.9090% |
| V3 80% + KD | 32 | 81.8222% | 90.9184% | 93.0458% | 94.9677% |
| V3 80%, KD 없음 | 27 | 79.1960% | 89.2312% | 92.1215% | 94.2048% |

80% KD-on − KD-off의 Test R@1 차이는 **+2.6262%p**, Val R@1 차이는 **+3.5226%p**다.
72%와 80% KD-on의 Test R@1 차이는 약 0.0147%p다.
전체 압축률과 재현 조건은 [실험 방법](METHOD.md)에 있다.

| 목표 | 완료 / 선택 epoch | Reduced Val R@1 | 파일 크기 (bytes) |
|---|---|---:|---:|
| 72% KD-on | 50 / 46 | 86.5142% | 10,655,695 |
| 76% KD-on | 34 / 24 | 85.5810% | 10,312,079 |
| 80% KD-on | 42 / 32 | 85.8176% | 9,967,823 |
| 80% KD-off | 37 / 27 | 82.2950% | 9,967,887 |

V3의 모든 조건에서 zero-pruned와 reduced 모델의 validation/test R@1/5/10/20이 동일했다.
파일 크기는 실제 직렬화된 checkpoint 크기다. 원본 training checkpoint는 optimizer를 포함하고
reduced checkpoint는 optimizer/scheduler를 제외하므로 두 파일의 크기 차이를 순수한 파라미터 감소로 해석하지 않는다.

## 보조 비교와 한계

같은 test에서 기존 epoch4 teacher는 R@1 80.6925%, V2 reduced KD-on/off는 78.5358%/75.9096%였다.
V2는 이전 teacher와 다른 종료 시점을 사용했고 KD-off의 triplet 계수도 1이었다.
따라서 V2와 최종 V3의 차이를 KD나 백본 하나의 효과로 분리하지 않는다.

- 최종 V3는 각 조건 seed 42 한 번의 결과다. 반복 seed와 통계적 유의성 검증은 하지 않았다.
- 학습 중 모델이 달라지므로 hard-negative 선택도 조건 간 달라질 수 있다.
- 독립 도시 데이터, latency, FLOPs, 에너지 사용량은 측정하지 않았다.
- teacher는 수동 종료 후 validation 최고점을 선택했다. 자동 early stopping 완료로 표시하지 않는다.

## 공개 결과 파일

- [final_comparison.csv](../results/final_comparison.csv): teacher 2개, V2 2조건, V3 4조건의 형태별 총 14행.
- [kd_ablation.csv](../results/kd_ablation.csv): 최종 80% KD-on/off의 zero-pruned/reduced 4행.
- [provenance.json](../results/provenance.json): checkpoint SHA·크기, 모델 선택·종료, 공개 파일 SHA와 원본 대응.
- `results/epochs/`: 각 모델의 전체 epoch CSV. loss는 해당 epoch 평균이며 Recall은 validation 지표다.
- `results/configs/`: 학습 조건. 로컬 출력 경로와 서버 전용 실행 인자는 제외했다.
- `results/evaluations/`: 실제 평가 JSON. 절대 작업 경로 접두사만 제거했다.

CSV/JSON의 Recall은 0–1 범위이고 문서 표는 %다. checkpoint 경로는 원본 파일의 저장소 상대 식별자이며
가중치 다운로드 링크가 아니다. 원시 CSV/JSON의 SHA와 공개 파일 SHA를 별도로 기록해 경로 정규화를 구분한다.

```bash
.venv/bin/python scripts/verify_results.py
```

이 검증은 파일 무결성, checkpoint/epoch/split 대응, 결과 간 일관성을 확인한다.
모델 추론에 의한 독립 재평가는 [재현 안내](REPRODUCING.md)의 checkpoint와 데이터가 필요하다.
