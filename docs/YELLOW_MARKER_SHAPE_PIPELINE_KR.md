# Yellow Marker Shape Pipeline (2D Planar)

## 개요

기존 ArUco 기반 EE pose 파이프라인은 유지한 상태에서, 노란색 원형 스티커를 영상에서 검출해 shape GT를 만드는 병렬 경로를 추가했습니다.

- 기존 경로: `recorder -> record_refiner -> aruco_reader -> aruco_refiner -> merger -> player`
- 신규 경로: `video -> marker_reader -> marker_refiner -> vision_shape_builder`
- 병합 확장: `record_refiner + vision_shape_builder -> merger.merge_marker_core`
- 데이터셋: `dataset/build_shape_residual_dataset.py`

## 단계별 실행

```bash
python -m marker_reader.app --video ./videos/run01.mp4 --out ./marker_reader/records/run01.csv --max-nodes 11 --preview
```

```bash
python -m marker_refiner.app --input ./marker_reader/records/run01.csv --out ./marker_refiner/results/run01_refined.csv --max-nodes 11
```

```bash
python -m vision_shape_builder.app --input ./marker_refiner/results/run01_refined.csv --out ./vision_shape_builder/results/run01_shape.csv --meters-per-pixel 0.001
```

```bash
python -m merger.merge_marker_core --record ./record_refiner/results/run01.csv --vision ./vision_shape_builder/results/run01_shape.csv --out ./merger/results/run01_merged_shape.csv --sync-tol-sec 0.05
```

```bash
python -m dataset.build_shape_residual_dataset --input ./merger/results/run01_merged_shape.csv --out ./dataset/run01_shape_residual.npz
```

## CSV 컬럼 요약

### `marker_reader` 출력

- `frame_idx`, `t_video_sec`, `n_detected`, `mean_conf`, `validity`, `sync_dt_sec`
- `c{i}_x`, `c{i}_y`, `c{i}_area`, `c{i}_circularity`, `c{i}_conf`

### `marker_refiner` 출력

- `frame_idx`, `t_video_sec`, `valid_shape`, `n_nodes`, `mean_conf`, `confidence`, `validity`, `sync_dt_sec`
- `node{i}_px_x`, `node{i}_px_y`, `node{i}_conf`

### `vision_shape_builder` 출력

- `frame_idx`, `t_video_sec`, `valid_shape`, `n_nodes`, `mean_conf`, `confidence`, `validity`, `sync_dt_sec`
- `node{i}_x_m`, `node{i}_y_m`, `node{i}_z_m`
- `joint{j}_deg`
- `ee_x_m`, `ee_y_m`, `ee_z_m`

### `merger.merge_marker_core` 출력

- 시간 동기화: `t_recorder`, `t_vision`, `dt_vision`, `sync_dt_sec`, `sync_tol_sec`
- 유효성: `valid_shape`, `valid_ee`, `valid_node{i}`, `imu_fresh`
- recorder + record_refiner 컬럼
- `vision_*` 컬럼
- nominal 컬럼: `nominal_node*`, `nominal_joint*`, `nominal_ee_*`
- residual 컬럼: `res_node*`, `res_ee_*`

## 알고리즘 메모

- `marker_reader`: HSV threshold + contour filtering(면적, circularity, aspect ratio)
- `marker_refiner`: 저신뢰 제거, base-to-tip 정렬, 이전 프레임 NN 추적, 점프 패널티, 짧은 결측 보간
- `vision_shape_builder`: pixel->metric 변환(`--meters-per-pixel` 또는 `--homography-json`), link/joint/EE 계산
- `merge_marker_core`: recorder timeline 기준 nearest-neighbor 동기화, 허용 오차 밖이면 vision을 `NaN` 처리

## 가정과 한계

- 현재는 단일 카메라 기반 2D planar shape 가정입니다.
- 깊이 정보가 없어서 out-of-plane 변형은 반영되지 않습니다.
- 노란 스티커는 ID가 없으므로 node swap 리스크가 남아 있습니다.
- segment boundary, EE에는 다른 색상 또는 별도 식별 마커를 쓰는 방식이 더 안전합니다.

