# Sơ đồ và cấu trúc toàn hệ thống MOT

Thiết kế đích theo [PROJECT_PLAN.md](../PROJECT_PLAN.md), cập nhật 18/11/2026. Phạm vi: một camera, người đi bộ, YOLO26s Detect + TrackTrack.

Video đã ghi và chuỗi ảnh được xử lý bằng CLI. Dịch vụ camera nhận luồng RTSP và cung cấp giao diện xem trực tiếp qua Caddy trên mạng LAN. Xem [hướng dẫn triển khai camera](../deploy/camera/README.md).

```mermaid
flowchart LR
    C["CLI · video hoặc ảnh MOT"] --> P["Pipeline · preset quality/fast"]
    P --> D["Đọc frame → YOLO26 Detect"]
    D --> T["TrackTrack · GMC có bảo vệ · ID"]
    T --> O["Vẽ → MP4 + MOT TXT + JSONL + summary"]
    O --> V["Batch · FFmpeg xuất preview H.264"]
    O --> E["TrackEval · chỉ khi có GT"]
    G["GT · MOT17/MOT20"] --> E
    E --> M["HOTA/IDF1/FN/FP/IDSW + phân tích lỗi"]
```

Tệp mới của V2:

- `src/mot/tracking/motion.py`: phát hiện frame thiếu tín hiệu/chuyển cảnh mạnh, kiểm tra warp GMC và reset track không dùng lại ID.
- `src/mot/tracking/replay.py` và `src/mot/evaluation/failure_analysis.py` hỗ trợ tìm tham số từ cache và giải trình FN/FP.
- `src/mot/app/` đọc luồng RTSP, theo dõi và phục vụ giao diện camera.
- `configs/profiles/`, `configs/runtime/`: hai preset và tùy chọn bảo vệ GMC/số luồng/capture. Model/tracker V2 có tệp riêng; các tệp gốc V1 được giữ nguyên.
- `scripts/experiments/` chạy và chọn cấu hình trên tập phát triển, đo tốc độ và đóng băng cấu hình.
- `scripts/scoring/` chấm GT, phân tích lỗi và đối chiếu sáu tiêu chí.
- `scripts/web_tools/` khởi động dịch vụ camera và kiểm tra bằng RTSP mô phỏng.
- `outputs/improvements/`: baseline/candidate, tìm tham số, regression, benchmark, cấu hình đóng băng, kết quả MOT20 và nghiệm thu.

## 1. Luồng từ input đến output

```mermaid
flowchart TD
    IN["INPUT: video hoặc chuỗi ảnh MOT17/MOT20"] --> ENTRY["Lệnh chạy CLI"]
    DATA["Cấu hình dữ liệu: nguồn, sequence, split"] -.-> ENTRY
    ENTRY --> READ["Đọc lần lượt từng khung hình và FPS"]

    subgraph LOOP["Xử lý từng khung hình"]
        READ --> DET["YOLO26s Detect: phát hiện người, khung bao, độ tin cậy"]
        DET --> TRACK["TrackTrack: ghép đối tượng và giữ ID"]
        MEMORY["Bộ nhớ tracker: ID và chuyển động từ các frame trước"] --> TRACK
        TRACK --> MEMORY
        TRACK --> DRAW["Vẽ khung bao, ID và quỹ đạo ngắn"]
        TRACK --> EXPORT["Lưu vị trí, ID và độ tin cậy theo frame"]
    end

    WEIGHTS["Trọng số YOLO26s Detect"] -.-> DET
    CFG["Cấu hình TrackTrack, ReID, bù chuyển động camera"] -.-> TRACK
    DRAW --> VIDEO["OUTPUT: video kết quả"]
    EXPORT --> TXT["OUTPUT: tracks.txt theo định dạng MOTChallenge"]
    LOOP --> SUMMARY["OUTPUT: summary.json với cấu hình và thống kê lần chạy"]

    TXT --> EVAL["TrackEval: đối chiếu kết quả với nhãn chuẩn"]
    GT["Nhãn chuẩn GT: chỉ dùng khi đánh giá"] --> EVAL
    EVAL --> METRICS["OUTPUT: HOTA, IDF1, số lần đổi ID"]
    EVAL --> ERRORS["OUTPUT: lỗi từng frame và 5 clip minh họa"]
    ERRORS --> REPORT
    METRICS --> REPORT["Báo cáo chất lượng, tốc độ và các lỗi tracking"]
    SUMMARY --> REPORT
    VIDEO --> PREVIEW["Batch: tệp preview H.264"]
```

- YOLO26 trả lời người đang ở đâu; TrackTrack nối người đó với ID trong lịch sử.
- Trạng thái tracker được giữ giữa các frame và reset khi đổi video.
- Hai khối detection/tracking là hai nhiệm vụ logic. Mã hiện dùng một lệnh `model.track(..., persist=True)` của Ultralytics để thực hiện cả hai; không cần gọi detector hai lần cho mỗi frame.
- Nhãn chuẩn không đi vào bước dự đoán ID. Video người dùng không có nhãn vẫn chạy được tracking, nhưng không chấm HOTA/IDF1.
- FPS được đo trong pipeline sau warm-up detector, tính đọc frame, tracking, vẽ và ghi kết quả/video. `summary.json` ghi phạm vi đo, phần cứng và độ trễ; TrackEval chấm chất lượng từ tệp MOT.
- Đánh giá chỉ dùng cửa sổ frame đã chạy, sao chép GT/metadata vào thư mục kết quả và bật tiền xử lý pedestrian. Dữ liệu gốc không bị chỉnh sửa.

## 2. Cấu trúc folder/file

Cây dưới đây liệt kê các tệp trong `scripts/`, `src/` và những đầu vào, đầu ra chính.

```text
MOT/
├── README.md                       # Cài đặt và các lệnh chạy
├── PROJECT_PLAN.md                 # Mục tiêu, các bước và tiêu chí hoàn thành
├── requirements.txt                # Phiên bản thư viện đã khóa cho môi trường CPU
├── .gitignore                      # Bỏ qua dữ liệu, trọng số, kết quả và cache
│
├── docs/
│   ├── ARCHITECTURE.md              # Sơ đồ và định nghĩa cấu trúc này
│   └── REPORT.md                   # Baseline, cải thiện và tập giữ lại
│
├── configs/
│   ├── datasets/
│   │   └── mot17.json              # Đường dẫn dữ liệu, sequence chạy thử và split
│   ├── models/
│   │   ├── yolo26s.json            # Model mặc định, nguồn/SHA và cấu hình inference
│   │   └── yolo26s_baseline.json   # Model/inference baseline bước 4 được giữ nguyên
│   ├── experiments/
│   │   ├── mot17_step5.json        # Cửa sổ phát triển, thứ tự và thay đổi từng phép thử
│   │   ├── mot17_step5_confirmation.json # Kiểm tra thêm, chọn và đóng băng trước đánh giá cuối
│   │   └── mot17_selected.json     # Quyết định chọn mặc định, metric phát triển và checksum
│   └── trackers/
│       ├── tracktrack.yaml         # Tham số tracking mặc định sau bước 5
│       └── tracktrack_baseline.yaml # Tham số baseline bước 4 được giữ nguyên
│
├── src/
│   └── mot/
│       ├── __init__.py
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── detection.py        # Chạy YOLO riêng, lưu detection
│       │   ├── pipeline.py         # Chạy YOLO + TrackTrack, xuất video và track
│       │   ├── motion.py           # Bảo vệ GMC, xử lý frame đen và chuyển cảnh
│       │   ├── replay.py           # Chạy lại TrackTrack từ cache
│       │   ├── sources.py          # Đọc video hoặc chuỗi ảnh MOT
│       │   ├── mot_io.py           # Xuất MOT TXT và chuẩn hóa tọa độ
│       │   ├── visualization.py    # Vẽ khung bao, ID và quỹ đạo
│       │   └── smoke.py            # Chạy mẫu MOT17 để kiểm tra môi trường
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py          # Gọi TrackEval, lưu điểm theo sequence và gộp
│       │   ├── diagnostics.py      # Đối chiếu lỗi CLEAR và cắt clip
│       │   └── failure_analysis.py # Phân tích FN/FP từ detection và GT
│       └── app/
│           ├── __init__.py
│           ├── camera.py           # Đọc RTSP và cập nhật tracking
│           ├── camera_web.py       # API trạng thái, ảnh và giao diện camera
│           ├── camera_ui/          # index.html, camera.js, camera.css
│           └── video.py            # Chuyển preview sang H.264
│
├── scripts/
│   ├── setup/
│   │   ├── check_setup.py          # Kiểm tra môi trường và dữ liệu
│   │   └── smoke_test.py           # Chạy ngắn trên MOT17
│   ├── inference/
│   │   ├── run_detection.py
│   │   ├── run_tracking.py
│   │   └── run_mot17.py            # Chạy batch, resume và xuất MOT TXT theo từng hậu tố
│   ├── scoring/
│   │   ├── evaluate.py
│   │   ├── evaluate_external.py    # Chấm MOT20 từ cấu hình đã đóng băng
│   │   ├── analyze_failures.py
│   │   └── verify_improvements.py
│   ├── experiments/
│   │   ├── capture_diagnostics.py
│   │   ├── search_tracking.py
│   │   ├── tune_tracking.py
│   │   ├── benchmark_profiles.py
│   │   └── freeze_profiles.py
│   └── web_tools/
│       ├── serve_camera.py
│       └── verify_camera.py
│
├── tests/
│   ├── test_detection.py           # Kiểm tra nguồn ảnh/video, confidence và export
│   ├── test_tracking.py            # Vòng đời ID, frame trống, quỹ đạo và reset nguồn
│   ├── test_evaluation.py          # MOT, metric, GT đoạn ngắn, distractor và clip
│   ├── test_tuning.py              # Chặn tập giữ lại, một thay đổi và quy tắc xếp hạng
│   └── test_batch.py               # Đủ train/test, ảnh trùng mọi frame và điều kiện dùng lại run
│
├── data/
│   ├── MOT17/                      # Dataset gốc: ảnh, metadata, nhãn chuẩn
│   ├── MOT17Labels/                # Bộ nhãn đã tải riêng, được giữ nguyên
│   └── MOT20/                      # Dữ liệu cảnh đông dùng để đánh giá ngoài MOT17
│
├── weights/
│   └── yolo26s.pt                  # Trọng số Detect, không chứa mã nguồn
│
├── outputs/
│   ├── checks/
│   │   └── setup_report.json       # Kết quả kiểm tra môi trường và dữ liệu
│   ├── runs/
│   │   ├── detect_mot17_09/         # Detection bước 2: video, JSONL, preview, summary
│   │   ├── track_mot17_09/          # Kết quả bước 3 trước khi thêm đo tốc độ/MOT
│   │   ├── baseline_mot17_09/       # Chạy lại ở bước 4; tương tự baseline_mot17_04
│   │   │   ├── tracking.mp4         # Video có ID và quỹ đạo
│   │   │   ├── tracks.jsonl         # Vị trí/ID/độ tin cậy của từng frame
│   │   │   ├── summary.json         # Cấu hình và thống kê lần chạy
│   │   │   └── tracks.txt           # Kết quả MOTChallenge, frame/tọa độ gốc 1
│   │   └── smoke_test/             # Hiện có; mỗi lần chạy đầy đủ sẽ dùng tên riêng
│   │       ├── preview.mp4         # Video có khung bao và ID
│   │       ├── frame_000001.jpg    # Ảnh preview đầu đoạn
│   │       ├── frame_000100.jpg    # Ảnh preview cuối đoạn mặc định 100 frame
│   │       └── summary.json        # Cấu hình, trọng số và thống kê lần chạy
│   ├── experiments/                # Kế hoạch, cấu hình, run và metric từng phép thử
│   ├── mot17_full/
│   │   ├── manifest.json           # Tiến độ, SHA ảnh, xuất xứ run dùng lại và thống kê
│   │   ├── configs/                # Snapshot model/tracker dùng chung cho cả lượt chạy
│   │   ├── train/                  # 7 sequence không lặp; mỗi sequence có video/MOT/JSONL
│   │   ├── test/                   # 7 sequence không lặp; không có GT cục bộ
│   │   ├── motchallenge/           # TXT theo cả 3 hậu tố, train.zip và test.zip
│   │   └── evaluation_train/       # Metric toàn bộ train, gồm cả phát triển và giữ lại
│   └── evaluations/
│       └── baseline_mot17/
│           ├── metrics.json        # Metric, FPS, phần cứng và phạm vi frame/split
│           ├── inputs/             # GT/metadata sao chép và prediction dùng để chấm
│           ├── trackeval/          # Summary và CSV chi tiết của evaluator
│           ├── *_errors.json       # FN/FP/IDSW từng frame
│           └── error_clips/        # 5 video/ảnh lỗi và index.json
│
└── .runtime/                       # Cache và thiết lập tạm của thư viện
```

## 3. Trách nhiệm và ranh giới

- **`scripts/` là cửa vào:** nhận tham số dòng lệnh, gọi phần xử lý phù hợp. `run_tracking.py` gọi pipeline; `smoke_test.py` gọi bài kiểm tra ngắn trong `smoke.py`.
- **`src/mot/tracking/pipeline.py` điều phối:** nạp model, warm-up, dùng bộ đọc nguồn chung, giữ TrackTrack xuyên suốt nguồn, ghi video/JSONL/MOT và đo tốc độ. Mỗi lần gọi tạo model/tracker mới, tránh chia sẻ ID giữa các video.
- **`detection.py` chạy bước 2:** gọi `model.predict`, chỉ lấy người, kiểm tra tọa độ ảnh gốc, lưu toàn bộ detection và video hiển thị. Không tạo tracker hay dùng GT.
- **`sources.py` cung cấp frame:** ảnh và thông tin FPS/kích thước, không dự đoán đối tượng.
- **`visualization.py` phục vụ hiển thị:** vẽ từ kết quả tracking, giữ tối đa 30 frame lịch sử và không nối đường qua frame thiếu quan sát. Không thay đổi ID để làm video đẹp hơn.
- **`mot_io.py` chuẩn hóa dữ liệu:** ghi `frame,id,left,top,width,height,confidence,-1,-1,-1`; chuyển `xyxy` gốc 0 sang MOT `xywh` gốc 1 và kiểm tra thứ tự frame, ID, kích thước box. Frame không có track vẫn phải có bản ghi JSONL.
- **`metrics.py` chấm kết quả** bằng TrackEval sau khi kiểm tra cấu hình và chuẩn bị GT đúng số frame. Kết quả gồm điểm từng sequence và điểm gộp.
- **`tune_tracking.py` thử tham số:** chỉ dùng các đoạn phát triển đã khai báo, mỗi biến thể thay một tham số của cấu hình cha. Lưu snapshot cấu hình và metric trước khi chọn cấu hình cho phép thử tiếp theo. Chế độ `--finalize-study` kiểm tra trên các đoạn phát triển còn lại, gộp điểm để chọn, lưu cấu hình/checksum rồi mới chạy tập giữ lại; không dùng điểm giữ lại để thay lựa chọn.
- **`diagnostics.py` giải thích lỗi:** dùng dữ liệu đã tiền xử lý, ghép theo CLEAR ở IoU 0.5 và đối chiếu tổng lỗi với evaluator trước khi tạo clip. Nhãn G/T trong clip là ID đã chuẩn hóa của TrackEval.
- **`video.py` tạo preview:** chuyển video tracking sang H.264 bằng FFmpeg cho batch. FPS pipeline không tính thời gian chuyển codec.
- **`run_mot17.py` điều phối toàn bộ dữ liệu:** kiểm tra SHA-256 mọi ảnh giữa DPM/FRCNN/SDP, chạy mỗi video một lần với cấu hình snapshot hoặc dùng lại run đầy đủ phù hợp. Sau khi xác minh số frame và xuất H.264, lưu kết quả theo sequence, xuất TXT cho cả ba tên và chấm toàn bộ train. Các bản hậu tố dùng chung kết quả vì ảnh đã được xác nhận giống hệt nhau; pipeline không đọc public detections của dataset.
- **`configs/` mô tả cấu hình và nguồn model:** detection/tracking nạp `models/yolo26s.json` để kiểm tra trọng số và lấy inference; tracking còn nạp `trackers/tracktrack.yaml`. Ngưỡng hiển thị trong cấu hình model chỉ dùng cho detection. Lệnh smoke test vẫn dùng các giá trị riêng trong mã chạy thử.
- **`data/` và `weights/` là đầu vào tải về; `outputs/` là dữ liệu sinh ra; `docs/` là tài liệu đọc và duy trì cùng mã nguồn.**

## 4. Gắn với kế hoạch triển khai

1. **Bước 1:** `check_setup.py`, cấu hình dữ liệu/model và `smoke_test.py` — đã chạy được.
2. **Bước 2–3:** đã có detection, nguồn đọc chung, tracking đầy đủ và quỹ đạo qua `pipeline.py`, `visualization.py`, `run_tracking.py`.
3. **Bước 4:** đã có xuất MOT, đánh giá, đo tốc độ và 5 clip lỗi; kết quả trong `docs/REPORT.md`.
4. **Bước 5:** đã thử 12 cấu hình, kiểm tra/chốt trên tập phát triển và đánh giá hai video giữ lại bằng `tune_tracking.py`; mặc định đã cập nhật. Giữ cấu hình bước 4 tại `yolo26s_baseline.json` và `tracktrack_baseline.yaml`; quyết định chọn trong `mot17_selected.json`.
5. **Bước 6:** kết quả kiểm chứng nằm trong `docs/REPORT.md`. CLI xuất video và tệp tracking để xem trực tiếp từ thư mục kết quả.

Không tạo module rỗng cho các phần dự kiến. Các lệnh hiện có:

```bash
conda activate mot
python scripts/setup/check_setup.py
python scripts/setup/smoke_test.py --frames 100
python scripts/inference/run_detection.py --frames 100 --output outputs/runs/detection_demo
python scripts/inference/run_tracking.py --frames 100 --output outputs/runs/tracking_demo
python scripts/scoring/evaluate.py --runs outputs/runs/tracking_demo --output outputs/evaluations/tracking_demo
```
