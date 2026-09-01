# 코드 구조와 이전 구현 대응

## 실행 흐름

1. `cluster`: CNN local descriptor 50,000개를 표본화하고 Faiss K-means로 NetVLAD centroid를 만든다.
2. `train`: 전체 train descriptor cache를 생성하고 hard-positive/hard-negative triplet을 채굴한다.
3. student의 triplet loss와 teacher/student soft-assignment KD loss를 함께 역전파한다.
4. `--pruning`이 지정된 경우 MobileNetV2 마지막 projection output을 지정 비율로 zeroing한다.
5. `reduce`: zero filter를 실제 채널에서 제거하고 NetVLAD assignment convolution과 centroid 차원도 함께 줄인다.
6. `test`: database/query descriptor를 생성하고 Faiss L2 검색으로 Recall@N을 계산한다.

## 이전 파일 대응

| 이전 코드 | 새 코드 |
|---|---|
| `main_for_KD.py`, `args.py` | `src/robot_compression/cli.py`, `config.py` |
| `trainer.py` | `src/robot_compression/training/engine.py` |
| `netvlad.py` | `src/robot_compression/models/netvlad.py` |
| `pittsburgh.py` | `src/robot_compression/data/pittsburgh.py` |
| `utils/pruning/common.py` | `src/robot_compression/compression/mobilenet.py` |
| `reducing.ipynb` | CLI의 `reduce` 모드 |

## 수정된 주요 오류

- student KD logits를 `no_grad` 밖에서 계산해 KD gradient가 student로 흐르게 했다.
- 단, 구버전은 student logits까지 `no_grad` 안에서 계산해 실제 KD gradient가
  없었다. 서로 독립적으로 생성한 teacher/student K-means cluster 번호도 정렬돼
  있지 않으므로 현재 soft-assignment KD는 구버전 재현값으로 간주하지 않는다.
  2026-08-31 재현 비교는 `--kd-weight 0`의 triplet + pruning으로 수행했다.
- 프루닝을 `--pruning` 사용 시에만 실행하고 `--pruning-ratio` 값을 적용한다.
- test 시 임의의 `reduced_model.pth`를 강제로 읽지 않고 명시된 checkpoint만 사용한다.
- 구버전 `pretrained=True` 조건을 재현하기 위해 MobileNetV2 backbone은
  `IMAGENET1K_V1`으로 고정하고 centroid 파일명에도 해당 revision을 기록한다.
- 데이터, teacher, centroid, checkpoint가 없으면 조용히 임의 초기화하지 않고 즉시 오류를 낸다.
- 물리 GPU 2와 논리 `cuda:0`의 매핑을 하나로 통일했다.
- 전체 모델 pickle 대신 구조 메타데이터와 `state_dict`를 저장한다.
- 구조 축소 checkpoint에서는 shape가 맞지 않아 재개할 수 없는 원본
  optimizer/scheduler 상태를 제외한다.

26% pruning을 30 epoch 수행한 실제 구조 축소 결과와 모든 비교 실험은
[`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)에 정리돼 있다.
