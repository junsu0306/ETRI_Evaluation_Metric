# MobileNetV2 KD-off 대조실험

2026-09-08, 시간대 Asia/Seoul.

## 최종 상태: 사용자 요청으로 종료

2026-09-08 21:40:26 KST에 KD-on/KD-off 학습 부모에 SIGINT를 전달했고,
21:40:31 이후 부모 및 소속 worker가 모두 종료됐음을 확인했다.
KD-on은 63 epoch, KD-off는 19 epoch까지 저장됐다. 진행 중이던 64/20 epoch는
미완료로 제외한다. 재실행이나 추가 백그라운드 작업은 예약하지 않았다.

R@1 최고 checkpoint는 KD-on epoch 37 (82.0978%), KD-off epoch 3 (79.0089%)다.
결과는 validation이며 student test/구조 축소 후 Recall은 미평가다.
[상세 최종 보고서](KD_ABLATION_FINAL_20260908.md)에 구현, 전체 epoch 및 한계를 정리했다.
아래 내용은 시작 시점의 설정과 검증 기록으로 보존한다.

## 목적과 비교 대상

사용자 요청에 따라 실행 중인 MobileNetV2 15% pruning 실험의 설정을 복제하고
`--kd-weight`만 `0.5`에서 `0`으로 변경한 대조군을 새로 시작했다.
기존 KD-on checkpoint를 이어받지 않고 같은 ImageNet V1 초기 가중치와
NetVLAD centroid에서 epoch 1부터 학습한다.

| 항목 | KD-on | KD-off |
|---|---|---|
| Run | `20260907_181321_mobilenetv2_netvlad_prune0p15_kdrel0p5` | `20260908_112757_mobilenetv2_netvlad_prune0p15` |
| 시작 시 확인한 PID | `1706268` | `2583716` |
| 물리 GPU | 2 | 2 |
| KD weight | 0.5 | 0 |
| Loss | `0.5 * triplet + 0.5 * KD` | `triplet` |

두 `config.json`을 프로그램으로 비교해 차이가 `kd_weight` 한 항목뿐임을
확인했다. 현재 CLI는 `(1 - kd_weight) * triplet + kd_weight * KD`를 사용하므로
KD-off에서는 triplet 계수도 0.5에서 1로 바뀐다. 따라서 이 결과는 기존 CLI의
KD weight 0 대조군이며, triplet 계수를 0.5로 고정한 KD 항 제거 실험은 아니다.
성능 차이를 해석할 때 이 loss scale 차이를 함께 명시해야 한다.

## 유지한 조건

- Pittsburgh30k train/val/test, MobileNetV2 ImageNet1K V1, NetVLAD 64 clusters.
- Seed 42, query group batch 1, cache batch 48, workers 8.
- 최대 100 epoch, patience 30. 현재 코드는 R@5 미갱신 횟수가 30을 초과하면
  조기 종료한다.
- SGD: learning rate 1e-4, momentum 0.9, weight decay 1e-3.
- StepLR: 5 epoch마다 learning rate에 0.5를 곱함.
- Margin 인자 0.1, 실제 TripletMarginLoss margin은 sqrt(0.1).
- Pruning 명시 활성화, 마지막 projection 출력 채널 pruning ratio 0.15.
- Temperature 0.1 인자 유지. KD-off에서는 KD 계산에 사용되지 않음.
- Validation R@5로 `best.pt`, R@1으로 `best_r1.pt`, 재개용 `latest.pt` 저장.
- Teacher checkpoint 인자도 유지해 모델 생성에 따른 난수 소비 순서를 맞춤.
  KD-off에서는 teacher descriptor 추론 및 KD loss 계산을 수행하지 않는다.
- GPU와 CPU thread 관련 환경변수는 기존 학습 프로세스의 설정을 유지했다.

## 실행과 산출물

- Run: `artifacts/runs/20260908_112757_mobilenetv2_netvlad_prune0p15`.
- 상세 실행 인자와 환경: 해당 run의 `config.json`, `run_metadata.json`.
- 로그: `artifacts/logs/20260908_112751_mobilenetv2_prune0p15_kd_off_control.log`.
- 실행 시각: 2026-09-08 11:27:51 KST; run 생성: 11:27:57 KST.
- 별도 세션으로 실행하고 stdin을 `/dev/null`, stdout/stderr를 로그 파일에
  연결했다. 시작 후 PPID=1, SID=2583716을 확인했다.
- 학생 feature cache는 run 이름별 경로를 사용하므로 기존 학습과 충돌하지 않는다.
- 두 학습이 GPU 2를 공유하므로 경과 시간은 성능 비교 지표로 사용하지 않는다.
- 초기 상태: 첫 feature cache 생성 완료 후 epoch 1의 첫 query subset에서
  `train 49/915`까지 진행 확인. 아직 완료 epoch/Recall 결과 없음.

## 재현성 식별자

Git commit: `baf5a0ce78220af958591798757ace349669fa45` (기존 미커밋 변경 포함).
이번 실행에서는 학습 소스를 수정하지 않았다.

SHA-256:

- MobileNetV2 V1 centroid: `c06c8b180ec8a57ffafe2b4a5e44eb80c7fe6d1a8bb84fbe59930832fc0463ef`.
- ResNet18 seed 123 teacher best: `bbab64a55eede03cc27ccd89cd764146adf8a4f515077f5f40b0292a276ee06c`.
- `src/robot_compression/cli.py`: `df390a5237fbff04a5b797b5556add304d74d23f8f420885467ef640544470ac`.
- `src/robot_compression/training/engine.py`: `6254622e27c0814ca2144cf047cba78e2474876beb63279d77c28feb173122c5`.

## 실행 전 검증

- `.venv/bin/python -m pytest`: 11 passed, 3 SWIG deprecation warnings, 4.95초.
- `.venv/bin/python -m robot_compression.cli check`: 성공, PyTorch 2.7.1+cu128,
  CUDA 사용 가능, physical GPU 2, val DB 10,000 / query 7,608.
- CPU smoke: MobileNetV2, 64 clusters, pruning ratio 0.15; 성공.
  Descriptor 20,480 → 17,408, 유지 채널 272, KD gradient norm 0.00000138.
- 검증용 CPU 실행에만 OMP/MKL/OpenBLAS thread 수 1 및 CUDA 비활성화를 적용했다.
  학습 실행은 위의 기존 프로세스 환경을 사용한다.
- 시작 직전 `scripts/check_resources.sh`: available RAM 471 GiB,
  디스크 약 2.0 TiB, GPU 2 free 18,022 MiB. 규칙의 최소 가용량 충족.
- 새 프로세스 시작 후 GPU 2 총 사용량 9,082 MiB, free 13,515 MiB 확인.
- 학습 배치 진행 중 재확인: 기존/대조군 프로세스 모두 실행 중,
  GPU 2 총 사용량 9,102 MiB, free 13,495 MiB. `git diff --check` 통과.
