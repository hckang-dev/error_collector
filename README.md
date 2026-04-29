# error_collector

<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/f6cf547e-ddc4-4b44-a16f-272b722b2e9a" />

(흐린 마커들은 이론상 위치, 원색 RGB 마커들은 aruco_reader가 읽어낸 위치입니다)

## 파일 흐름

```text
recorder ─> record_refiner ────────────────┐
aruco_reader ─> aruco_refiner ───────────┐ │
video ─> marker_reader ─> marker_refiner ─> vision_shape_builder ─┤ └─> merger ─> player
                                           └───────────────────────┘
실행은 각 폴더 내의 app.py로
```

```text
1-A. recorder
   -> 모터 값, 로드 값, IMU full sensor fields 기록

1-B. record_refiner
   -> recorder CSV를 기반으로 모델상 ArUco 마커의 이론 위치/자세 계산

2-A. aruco_reader
   -> 동영상을 읽어 ArUco 마커 위치/자세 추출

2-B. aruco_refiner
   -> 영상 기반 ArUco 결과를 필터링하고 결측을 일부 보정

3. merger
   -> (Aruco 경로) record_refiner + aruco_refiner 병합
   -> (Shape 경로) record_refiner + vision_shape_builder 병합

4. player
   -> 병합 CSV를 읽어 Genesis에서 재생 및 시각화

5. dataset
   -> shape residual 학습용 NPZ 데이터셋 생성
```

## 역할 소개

### recorder

로봇 하드웨어에서 측정한 값을 기록합니다.

- 모터 제어값 `roll`, `seg1`, `seg2`
- 모터 부하값 `load1`, `load2`
- 두 개의 IMU full fields (timestamp, gravity, gyro, quaternion, accuracy, linear accel, accel, mag)

출력 CSV 예시는 다음과 같습니다(컬럼 일부 축약).

```text
t,roll,seg1,seg2,load1,load2,
imu1_timestamp,imu1_gvx,imu1_gvy,imu1_gvz,imu1_gx,imu1_gy,imu1_gz,
imu1_qw,imu1_qx,imu1_qy,imu1_qz,imu1_quatAcc,imu1_gravAcc,imu1_gyroAcc,
imu1_lax,imu1_lay,imu1_laz,imu1_ax,imu1_ay,imu1_az,imu1_mx,imu1_my,imu1_mz,
imu2_timestamp, ... , imu2_mz
```

### record_refiner

`recorder`가 저장한 모터 값을 바탕으로, 로봇 모델 기준 ArUco 마커들의 이론상 위치와 자세를 계산합니다.

- 입력: `recorder/records/*.csv`
- 출력: `record_refiner/results/*.csv`
- 추가되는 컬럼:

```text
mp{id}x, mp{id}y, mp{id}z,
mq{id}x, mq{id}y, mq{id}z, mq{id}w
```

즉, 실제 영상에서 본 마커 위치가 아니라, 현재 모터 상태라면 모델상 마커가 어디에 있어야 하는지를 계산해 붙입니다.

### aruco_reader

동영상을 읽어 ArUco 마커를 검출하고, 각 프레임에서 마커 위치와 자세를 기록합니다.

- 입력: 비디오 파일
- 출력: `aruco_reader/records/*.csv`
- 사용 마커:
  - sync marker: `DICT_5X5_50`
  - measurement marker: `DICT_4X4_50`

출력 CSV에는 카메라 기준 marker pose가 저장됩니다.

```text
t,
pCAMx,pCAMy,pCAMz,qCAMx,qCAMy,qCAMz,qCAMw,
p1x,p1y,p1z,q1x,q1y,q1z,q1w,
...
```

### aruco_refiner

`aruco_reader`가 만든 영상 기반 마커 추적 결과를 정제합니다.

- frame 간 갑작스러운 position jump 제거
- quaternion angle jump 제거
- 친구 마커 그룹 투표로 이상값 완화
- 짧은 결측 구간 보간

출력은 원래 ArUco pose 컬럼을 유지하면서 진단 컬럼을 추가합니다.

```text
valid{id},quality{id},reason{id},angle_jump{id},pos_step{id}
```

즉, 영상에서 검출한 값 중 튀는 값을 줄이고, 후속 병합과 재생에 더 안정적인 CSV를 만드는 단계입니다.

### merger

`record_refiner` 결과와 `aruco_refiner` 결과를 시간 기준으로 합칩니다.

- recorder 계열 타임라인을 기준으로 사용
- 가장 가까운 ArUco timestamp를 찾아 붙임
- 허용 오차는 ArUco frame 간격으로부터 half-frame 수준으로 추정

출력 CSV에는 다음 정보가 함께 들어갑니다.

- recorder 쪽 모터/로드/IMU 값
- record_refiner가 계산한 모델 기반 마커 pose
- aruco_refiner가 정제한 영상 기반 마커 pose

또한 `merge_marker_core.py`를 통해 `record_refiner`와 `vision_shape_builder` 결과를 시간 정렬 병합할 수 있습니다.

### player

`merger`가 만든 최종 CSV를 읽어 Genesis 환경에서 재생합니다.

- 로봇 URDF 로드
- recorder 기반 관절 상태 재생
- 영상 기반 ArUco 마커와 모델 기반 마커를 함께 시각화
실험 결과를 눈으로 비교하면서, 실제 검출값과 모델 계산값이 얼마나 맞는지 보는 용도입니다.

## 폴더 구성

```text
error_collector/
  recorder/         모터/IMU 기록
  record_refiner/   모터값 기반 이론 마커 pose 계산
  aruco_reader/     영상에서 ArUco marker pose 추출
  aruco_refiner/    ArUco 결과 필터링 및 보정
  marker_reader/    노란 스티커 contour 검출
  marker_refiner/   스티커 정렬/추적/보간 정제
  vision_shape_builder/ pixel shape -> metric shape 변환
  merger/           두 결과를 시간축 기준으로 병합
  player/           병합 결과 재생
  dataset/          shape residual NPZ 생성
  docs/             파이프라인 문서
```

## 입출력 경로

```text
recorder output          -> recorder/records
record_refiner input     -> recorder/records
record_refiner output    -> record_refiner/results

aruco_reader output      -> aruco_reader/records
aruco_refiner input      -> aruco_reader/records
aruco_refiner output     -> aruco_refiner/results

marker_reader output     -> marker_reader/records
marker_refiner input     -> marker_reader/records
marker_refiner output    -> marker_refiner/results
vision_shape_builder input  -> marker_refiner/results
vision_shape_builder output -> vision_shape_builder/results

merger recorder input    -> record_refiner/results
merger aruco input       -> aruco_refiner/results
merger vision-shape input -> vision_shape_builder/results
merger output            -> merger/results

player input             -> merger/results
dataset input            -> merger/results
dataset output           -> dataset/*.npz
```

## 설치

루트에서 다음처럼 설치하면 됩니다.

```bash
pip install -r requirements.txt
```

GUI 앱 실행에는 `tkinter`가 필요합니다. Ubuntu/Debian 계열에서는 보통 다음 패키지를 설치합니다.

```bash
sudo apt install python3-tk
```

## 실행 순서

### 1. recorder

```bash
cd recorder
python3 app.py
```

기록 결과는 `recorder/records`에 저장됩니다.

### 2. record_refiner

```bash
cd record_refiner
python3 app.py
```

정제 결과는 `record_refiner/results`에 저장됩니다.

### 3. aruco_reader

```bash
cd aruco_reader
python3 app.py
```

비디오를 열고 Detect 후 Export하면 `aruco_reader/records`에 저장됩니다.

### 4. aruco_refiner

```bash
cd aruco_refiner
python3 app.py
```

정제 결과는 `aruco_refiner/results`에 저장됩니다.

### 5. merger

```bash
cd merger
python3 app.py
```

병합 결과는 `merger/results`에 저장됩니다.

### 6. player

```bash
cd player
python3 app.py
```

필요하면 파일을 바로 지정해서 열 수 있습니다.

```bash
python3 app.py --file ../merger/results/testP.csv
```

## 신규: 노란 원형 마커 기반 shape 파이프라인

ArUco 파이프라인과 병렬로 노란색 스티커 기반 shape GT 경로를 추가했습니다.

- `node0~node10`: 도넛형 노란 원형 관절 스티커
- `marker_reader`는 모든 노드를 원형 마커로 동일하게 검출합니다.
- 원형은 도넛 내부 hole을 별도 마커로 세지 않도록 외곽 contour 기준으로 처리합니다.

- `marker_reader -> marker_refiner -> vision_shape_builder`
- `record_refiner + vision_shape_builder -> merger.merge_marker_core`
- `dataset/build_shape_residual_dataset.py`

상세 실행법과 컬럼 설명은 `docs/YELLOW_MARKER_SHAPE_PIPELINE_KR.md`를 참고하세요.

빠른 실행 예시는 다음과 같습니다.

```bash
python -m marker_reader.app --video ./videos/run01.mp4 --out ./marker_reader/records/run01.csv --max-nodes 11 --preview
python -m marker_refiner.app --input ./marker_reader/records/run01.csv --out ./marker_refiner/results/run01_refined.csv --max-nodes 11
python -m vision_shape_builder.app --input ./marker_refiner/results/run01_refined.csv --out ./vision_shape_builder/results/run01_shape.csv --meters-per-pixel 0.001
python -m merger.merge_marker_core --record ./record_refiner/results/run01.csv --vision ./vision_shape_builder/results/run01_shape.csv --out ./merger/results/run01_merged_shape.csv --sync-tol-sec 0.05
python -m dataset.build_shape_residual_dataset --input ./merger/results/run01_merged_shape.csv --out ./dataset/run01_shape_residual.npz
```

## 신규: frame+sensor raw dataset 수집

마커 검출/GT 생성 이전 단계로, 카메라 프레임과 센서값을 시간 동기화해 raw multimodal dataset을 만들 수 있습니다.

- 패키지: `frame_sensor_dataset`
- 출력 구조:

```text
datasets/run01/
  frames/
    frame_000000.jpg
    ...
  manifest.csv
  manifest.jsonl
```

- manifest 핵심 컬럼:
  - `frame_path`, `t_frame_wall`, `t_frame_mono`
  - `t_sensor_mono`, `dt_sensor_sec`, `sync_valid`
  - 모터/전류/전압/IMU 필드

비디오 + 센서 로그 동기화 예시:

```bash
python -m frame_sensor_dataset.collect_from_video \
  --video ./videos/run01.mp4 \
  --sensor-log ./recorder/records/run01.csv \
  --out-dir ./datasets/run01 \
  --sync-tol-sec 0.05 \
  --save-every-n 1 \
  --image-ext jpg
```

라이브 카메라 + 센서 스트림 예시:

```bash
python -m frame_sensor_dataset.collect_live \
  --camera-index 0 \
  --sensor-zmq tcp://127.0.0.1:5556 \
  --imu-port /dev/ttyACM0 \
  --out-dir ./datasets/run01 \
  --sync-tol-sec 0.05 \
  --save-every-n 1 \
  --image-ext jpg
```

## 데이터 해석 메모

- 위치 단위는 meter입니다.
- ArUco quaternion 순서는 `(x, y, z, w)`입니다.
- recorder 원본 IMU는 quaternion뿐 아니라 gravity/gyro/acc/mag/accuracy/timestamp 전체 필드를 포함합니다.
- player는 영상 기반 marker pose와 모델 기반 marker pose를 함께 읽어 비교 시각화합니다.
