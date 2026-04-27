# Aruco Refiner

Refines the wide-format CSV exported by `error_collector/aruco_reader`.

Input rows keep the existing shape:

```text
t,pCAMx,pCAMy,pCAMz,qCAMx,qCAMy,qCAMz,qCAMw,p1x,p1y,p1z,q1x,q1y,q1z,q1w,...
```

The refiner writes the same marker columns, rejects frame-to-frame outliers, and
adds diagnostics:

```text
valid{id},quality{id},reason{id},angle_jump{id},pos_step{id}
```

Run the GUI:

```bash
python3 app.py
```

Choose an input CSV from `error_collector/aruco_reader/records`, choose where to save
the refined CSV, adjust thresholds if needed, then click `Run Refine`.

Command-line usage is still available:

```bash
python3 -m error_collector.aruco_refiner.app input.csv output_refined.csv
```

Thresholds:

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

Voting:

- Friend groups are `(1,2,3)`, `(4,5,6)`, `(7,8,9)`, and `(10,11,12)`.
- A marker is checked against both its own last accepted pose and the current
  motion of its friends.
- If a marker jumps relative to its own past but at least `vote-min-support`
  markers in the friend group show consistent motion, the row is accepted with
  `reason=voted`.
- If a marker passes its own temporal check but disagrees with available
  friends, it is rejected with `reason=vote_mismatch`.

Coordinate convention:

- `pCAM=(0,0,0)` and `qCAM=(0,0,0,1)` represent the OpenCV camera origin and
  orientation.
- `p{id}` is the marker center position in the input CSV coordinate frame.
- `q{id}` is the marker orientation quaternion in `(x,y,z,w)` order in the same
  coordinate frame.
