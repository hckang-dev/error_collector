# error_collector

ArUco 영상, 센서 recorder CSV, 정제(refine), 병합(merge), Genesis player 재생까지 이어지는 실험 데이터 수집/검증 도구 모음입니다.

## 구성

```text
error_collector/
  aruco_reader/     영상에서 ArUco marker pose CSV를 생성
  aruco_refiner/    ArUco pose CSV의 튐값 제거, 투표 기반 검증, 짧은 결측 보간
  recorder/         roll, segment, load, IMU quaternion 기록
  merger/           recorder CSV와 refined ArUco CSV를 시간 기준으로 병합
  player/           병합 CSV를 Genesis에서 재생하고 marker pose를 시각화
```

각 단계의 파일 선택기는 앞 단계 결과 폴더를 기본으로 엽니다.

```text
aruco_reader output      -> aruco_reader/records
aruco_refiner input      -> aruco_reader/records
aruco_refiner output     -> aruco_refiner/results
recorder output          -> recorder/records
merger recorder input    -> recorder/records
merger aruco input       -> aruco_refiner/results
merger output            -> merger/results
player input             -> merger/results
```

## 설치

기본 Python 의존성:

```bash
pip install numpy opencv-python
```

player까지 사용할 경우:

```bash
pip install -r player/requirements.txt
```

GUI는 `tkinter`가 필요합니다. Ubuntu/Debian 계열에서는 보통 다음 패키지가 필요합니다.

```bash
sudo apt install python3-tk
```

## 실행 순서

## 실행 결과만 빠르게 보기

전체 pipeline을 다시 돌리지 않고 결과 재생만 보고 싶으면 player를 실행한 뒤 `testP.csv`를 열면 됩니다.

```bash
cd error_collector/player
python3 app.py
```

파일 선택 창에서 다음 파일을 선택합니다.

```text
error_collector/merger/results/testP.csv
```

CLI로 바로 열 수도 있습니다.

```bash
cd error_collector/player
python3 app.py --file ../merger/results/testP.csv
```

### 1. ArUco 영상 읽기

```bash
cd error_collector/aruco_reader
python3 app.py
```

- 비디오를 선택합니다.
- Detect를 실행합니다.
- Export하면 CSV가 `aruco_reader/records`에 저장됩니다.

ArUco reader 출력 포맷은 marker 위치와 quaternion을 그대로 저장합니다.

```text
t,
pCAMx,pCAMy,pCAMz,qCAMx,qCAMy,qCAMz,qCAMw,
p1x,p1y,p1z,q1x,q1y,q1z,q1w,
...
```

카메라 기준값은 고정입니다.

```text
pCAM = (0, 0, 0)
qCAM = (0, 0, 0, 1)
```

marker quaternion은 `(x, y, z, w)` 순서입니다.

### 2. Recorder 실행

```bash
cd error_collector/recorder
python3 app.py
```

recorder는 roll, segment, load, IMU quaternion 등을 CSV로 저장합니다.

```text
t,roll,seg1,seg2,load1,load2,
imu1_qw,imu1_qx,imu1_qy,imu1_qz,
imu2_qw,imu2_qx,imu2_qy,imu2_qz
```

출력은 `recorder/records`에 저장됩니다.

### 3. ArUco Refiner

```bash
cd error_collector/aruco_refiner
python3 app.py
```

파일 선택기는 `aruco_reader/records`를 기본으로 엽니다. 정제 결과는 `aruco_refiner/results`에 저장됩니다.

CLI도 사용할 수 있습니다.

```bash
python3 -m error_collector.aruco_refiner.app input.csv output_refined.csv \
  --max-angle-deg 75 \
  --max-position-step-m 0.10 \
  --reset-gap-rows 5 \
  --interpolate-gap-rows 2 \
  --vote-min-support 2 \
  --vote-max-delta-step-m 0.04 \
  --vote-max-delta-angle-deg 30
```

Refiner는 기존 marker pose 컬럼을 유지하고 진단 컬럼을 추가합니다.

```text
valid{id},quality{id},reason{id},angle_jump{id},pos_step{id}
```

주요 reason:

```text
first           첫 유효 pose
accepted        자기 과거 기준 통과
voted           자기 과거 기준으로는 튀었지만 친구 marker 투표로 통과
vote_mismatch   자기 과거 기준은 통과했지만 친구 marker들과 움직임이 맞지 않음
position_jump   위치 변화량 초과
angle_jump      quaternion 회전 변화량 초과
missing         pose 없음
interpolated    짧은 결측 보간
reset           긴 결측 뒤 재획득
```

#### 투표 기능

marker는 3개씩 친구 그룹을 이룹니다.

```text
(1,2,3), (4,5,6), (7,8,9), (10,11,12)
```

각 marker는 자기 자신의 마지막 accepted pose와 현재 pose를 비교하고, 동시에 같은 그룹 친구들의 현재 움직임과 비교합니다.

- 자기 과거 기준으로는 튀었지만 친구들과 같은 방향/크기로 움직였으면 `voted`로 통과합니다.
- 자기 과거 기준으로는 정상이어도 친구들과 혼자 다른 움직임이면 `vote_mismatch`로 reject합니다.
- 친구 정보가 부족하면 자기 과거 기준만 사용합니다.

기본 위치 jump 허용값은 `0.10 m`입니다. 즉 마지막 accepted pose 대비 10cm 초과로 튀면 의심합니다.

### 4. Merger

```bash
cd error_collector/merger
python3 app.py
```

파일 선택기는 다음 위치를 기본으로 엽니다.

```text
Recorder CSV       -> recorder/records
Refined ArUco CSV  -> aruco_refiner/results
```

recorder row를 기준 timeline으로 사용하고, 각 recorder timestamp에 가장 가까운 ArUco row를 붙입니다. 허용 오차는 ArUco CSV frame 간격에서 추론한 half-frame입니다.

출력은 `merger/results`에 저장됩니다.

merged CSV에서는 IMU quaternion을 사용하기 편한 순서로 재배열합니다.

```text
qIMU1x,qIMU1y,qIMU1z,qIMU1w
qIMU2x,qIMU2y,qIMU2z,qIMU2w
```

ArUco pose는 다음 형태로 유지됩니다.

```text
pCAMx,pCAMy,pCAMz,qCAMx,qCAMy,qCAMz,qCAMw
p{id}x,p{id}y,p{id}z,q{id}x,q{id}y,q{id}z,q{id}w
```

### 5. Player

```bash
cd error_collector/player
python3 app.py
```

파일 선택기는 `merger/results`를 기본으로 엽니다.

옵션:

```bash
python3 app.py --file /path/to/merged.csv
python3 app.py --urdf /path/to/other_robot.urdf
python3 app.py --gpu
```

player는 `q{id}x/y/z/w`를 `(x,y,z,w)` quaternion으로 읽습니다. 구형 `v{id}x/y/z` normal vector 컬럼도 fallback으로 읽을 수 있습니다.

marker 시각화:

- 17mm filled square marker plate
- marker local `+Z` normal stick
- normal stick 길이: 10mm

## 권장 Refiner 설정

일반적인 시작점:

```bash
--max-angle-deg 45 \
--max-position-step-m 0.10 \
--reset-gap-rows 5 \
--interpolate-gap-rows 2 \
--vote-min-support 2 \
--vote-max-delta-step-m 0.04 \
--vote-max-delta-angle-deg 30
```

느린 움직임/고정 실험에서 더 빡세게 잡을 때:

```bash
--max-angle-deg 30 \
--max-position-step-m 0.06 \
--vote-max-delta-step-m 0.025 \
--vote-max-delta-angle-deg 20
```

빠른 움직임이나 노이즈가 큰 영상에서 더 널널하게 잡을 때:

```bash
--max-angle-deg 60 \
--max-position-step-m 0.15 \
--vote-max-delta-step-m 0.06 \
--vote-max-delta-angle-deg 40
```

진단 컬럼을 보고 조정합니다.

```text
vote_mismatch가 너무 많음 -> vote-max-delta-step-m 또는 vote-max-delta-angle-deg 증가
position_jump가 너무 많음 -> max-position-step-m 증가
튀는 회전이 살아남음 -> max-angle-deg 감소
interpolated가 너무 많음 -> interpolate-gap-rows 감소
```

## 좌표와 quaternion convention

- 위치 단위는 meter입니다.
- marker quaternion은 `(x, y, z, w)` 순서입니다.
- Genesis 내부 link quaternion은 필요한 곳에서 `(w, x, y, z)`로 변환합니다.
- `pCAM=(0,0,0)`, `qCAM=(0,0,0,1)`은 OpenCV camera origin/orientation을 나타내는 고정 기준입니다.

## 주의사항

- Refiner는 새 `p/q` 포맷 CSV를 기준으로 동작합니다. 구형 `p/v` CSV는 player fallback에서는 읽을 수 있지만, refiner 대상은 아닙니다.
- Refiner는 `aruco_reader -> aruco_refiner -> merger -> player` 순서로 사용하는 것을 전제로 합니다.
- Genesis가 marker plate entity에서 solver 문제를 일으키는 환경이라면, marker plate 생성 방식을 debug-only fallback으로 바꿔야 할 수 있습니다.
