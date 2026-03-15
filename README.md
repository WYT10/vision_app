# vision_app

Lightweight USB camera probe and runtime tester for Linux USB cameras using:
- `v4l2-ctl` for advertised camera modes
- OpenCV 4.10 for real capture testing

## Layout

```text
vision_app/
├── CMakeLists.txt
├── README.md
├── vision_app.conf
├── inc/
├── src/
└── report/
```

## Features

- Normalized probe table: deduplicates repeated mode/FPS lines from `v4l2-ctl`
- Runtime FPS measurement with low-memory capture loop
- Append-only `test_results.csv`
- `probe_table.csv` export
- `latest_report.md` export
- Auto-creates report directories
- Keeps `main.c` as the entry file while compiling as C++

## Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake v4l-utils libopencv-dev
```

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Probe only

```bash
./vision_app --device /dev/video0 --probe-only
```

## Runtime test

```bash
./vision_app --device /dev/video0 --width 1280 --height 720 --fps 30 --fourcc MJPG --duration 10 --headless
```

## Recommended report files

When launched from `build/`, the default config writes to `../report/`:

- `../report/probe_table.csv`
- `../report/test_results.csv`
- `../report/latest_report.md`

## Notes

- `CAP_PROP_BUFFERSIZE = 1` is used to reduce stale-frame buildup.
- Preview can trigger desktop/runtime warnings on Pi sessions; use `--headless` for clean testing.
- The report schema already leaves room for future YOLO timing fields.
