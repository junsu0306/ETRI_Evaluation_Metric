# ETRI Robot Compression Evaluation

Pittsburgh30k 장소 인식 모델에서 **관계 기반 지식 증류(KD)**와 **구조적 채널 축소**의 효과를 평가합니다.
ResNet18 + NetVLAD64를 teacher로 사용하고, MobileNetV3-Large + NetVLAD64를
학습 가능 파라미터 기준 72%·76%·80% 압축 조건으로 비교했습니다.

학습, 실제 구조 축소, validation/test 평가와 80% KD-off 대조실험은 **2026-09-11 완료**했습니다.
이 저장소는 해당 최종 결과, 실행 코드, 재현 절차를 포함한 과제 제출용 공개 버전입니다.

## 최종 결과

학생 모델은 모두 **실제 구조 축소 후** 결과입니다. Checkpoint는 validation R@1으로 선택하며
test split은 선택이나 조기 종료에 사용하지 않았습니다.

| 모델 | 선택 epoch | Test R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| 공식 ResNet18 teacher | 14 | 85.5927% | 91.9748% | 93.9554% | 95.3492% |
| V3 72% + KD | 46 | 81.8369% | 90.3756% | 92.8404% | 95.0411% |
| V3 76% + KD | 24 | 81.7488% | 90.3462% | 92.7377% | 94.9090% |
| V3 80% + KD | 32 | 81.8222% | 90.9184% | 93.0458% | 94.9677% |
| V3 80%, KD 없음 | 27 | 79.1960% | 89.2312% | 92.1215% | 94.2048% |

- 80% 압축 조건에서 KD의 Test R@1 개선은 **+2.6262%p**입니다.
- 대조군의 triplet 계수도 0.5로 유지해 `0.5 × Triplet + 0.5 × KD`와 `0.5 × Triplet`을 비교했습니다.
- V3 모델의 구조 축소 전후 Recall@1/5/10/20은 validation/test 모두 동일했습니다.
- **80%는 teacher 대비 학습 가능 파라미터 감소율**입니다. 80% 모델의 실제 감소율은 trainable 80.0195%, total 13.1555%입니다.
- 해당 reduced 모델은 total/trainable **2,445,152/426,080개**, checkpoint **9,967,823 bytes**입니다.

조건별 seed 42 한 번씩의 결과이며, 통계적 유의성·추론 속도·FLOPs·에너지는 측정하지 않았습니다.
파라미터 감소를 같은 비율의 실행 속도 개선으로 해석하지 않습니다.
[전체 결과와 한계](docs/RESULTS.md), [원본 정밀도의 비교 CSV](results/final_comparison.csv)를 제공합니다.

## 설치와 빠른 확인

기준 환경은 **Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128**입니다.
버전은 [requirements.txt](requirements.txt)에 고정했습니다. 명령은 저장소 루트에서 실행합니다.

```bash
git clone https://github.com/junsu0306/ETRI_Evaluation_Metric.git
cd ETRI_Evaluation_Metric
./scripts/bootstrap_env.sh
cp .env.example .env
```

`.env`에서 사용 가능한 물리 GPU를 `CUDA_VISIBLE_DEVICES`로 지정합니다.
데이터·run·cache 경로는 기본적으로 저장소 루트를 기준으로 정해집니다.
데이터와 학습 가중치 없이도 CPU 테스트 및 결과 파일 검증을 실행할 수 있습니다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m robot_compression.cli smoke --cpu \
  --arch mobilenetv3-large --num-clusters 64 --pruning-ratio 0.49166666666666664
.venv/bin/python scripts/verify_results.py
```

## 재현과 제공 범위

데이터 다운로드, centroid 생성, teacher/student 학습, KD-off, 구조 축소와 test 명령은
[재현 안내](docs/REPRODUCING.md)에 순서대로 정리했습니다.

공개 Git에는 코드·테스트·문서와 작은 CSV/JSON 결과를 포함합니다.
데이터셋, ImageNet 가중치, 학습 checkpoint, centroid, cache, 로그는 포함하지 않습니다.
저장된 점수의 무결성 확인은 즉시 가능하며, **동일 checkpoint의 재평가에는 별도로 보관된 가중치가 필요**합니다.
새로 학습하면 환경과 난수 실행 차이로 수치가 달라질 수 있습니다.

## 문서와 결과 파일

| 파일 | 내용 |
|---|---|
| [실험 방법](docs/METHOD.md) | 모델, pruning 축, KD 목적함수, 학습·평가 조건 |
| [최종 결과](docs/RESULTS.md) | 전체 Recall, 압축률, 대조군, 해석 범위 |
| [재현 안내](docs/REPRODUCING.md) | 설치부터 평가까지의 명령과 필요한 입력 |
| [코드 구조](docs/ARCHITECTURE.md) | 모듈별 역할과 실행 흐름 |
| [데이터셋](docs/DATASET.md) | 다운로드, split, 디렉터리 구성 |
| [공개 버전 검증](docs/VALIDATION.md) | 테스트·CPU smoke·Git 추적 파일 검증 |
| [최종 비교 CSV](results/final_comparison.csv) | teacher·V2·V3의 축소 전후 지표 14행 |
| [KD 대조실험 CSV](results/kd_ablation.csv) | 80% KD-on/off의 축소 전후 비교 |
| [결과 출처](results/provenance.json) | 선택 epoch, SHA-256, 종료 사유, 원본/공개 파일 대응 |

`results/epochs/`, `results/configs/`, `results/evaluations/`에는 모델별 전체 epoch 지표,
학습 설정, 평가 JSON을 제공합니다. CSV/JSON의 Recall 단위는 **0–1**, 문서 표는 **%**입니다.

```text
src/robot_compression/  모델·데이터 로더·학습·pruning·CLI
scripts/               환경 구성·데이터 다운로드·자원 및 결과 검증
tests/                 KD gradient·구조 축소·checkpoint 선택 검증
docs/                  방법·결과·재현 문서
results/               공개용 CSV/JSON 실험 근거
```
