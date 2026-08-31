# Agent instructions

이 저장소에서 작업하는 모든 에이전트는 먼저 `RULE.md`를 읽고 준수한다.

1. `.venv` 외부에 Python 패키지를 설치하지 않는다.
2. 장시간 또는 GPU 작업 전에 `scripts/check_resources.sh`를 실행한다.
3. 기본 물리 GPU는 2번이며, 다른 사용자의 프로세스를 종료하지 않는다.
4. 데이터와 체크포인트는 Git에 넣지 않는다.
5. 구버전 파일은 삭제하지 않고 `trash/legacy`로 이동한다.
6. 코드 변경 후 테스트와 smoke check 결과를 기록한다.
7. 사용자 변경사항과 무관한 파일은 되돌리지 않는다.
