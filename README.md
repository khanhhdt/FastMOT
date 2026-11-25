# MOT: YOLO26 Detect + TrackTrack

Theo dõi người đi bộ trong video bằng YOLO26 Detect và TrackTrack.

## Camera IP trực tiếp

Chế độ live dùng YOLO26s OpenVINO FP32 + TrackTrack + ReID, với preset tracking 3 và 4 FPS; giao diện cập nhật ảnh nguồn tối đa 15 FPS để xem mượt hơn. Xem [hướng dẫn triển khai, chạy RTSP mô phỏng và nghiệm thu](deploy/camera/README.md).

## Xử lý video đã ghi bằng CLI

Chạy từ thư mục dự án và dùng thư mục đầu ra mới:

```bash
conda activate mot
python scripts/inference/run_tracking.py --source path/to/video.mp4 --profile quality --output outputs/runs/my_quality
python scripts/inference/run_tracking.py --source path/to/video.mp4 --profile fast --output outputs/runs/my_fast
```

`quality` là mặc định của CLI; các cờ `--model-config`, `--tracker-config`, `--runtime-config` ghi đè từng phần của preset. Cấu hình nằm trong `configs/profiles/`; bản nhanh và chất lượng vẫn cùng giải pháp YOLO26 Detect + TrackTrack. Pipeline có bảo vệ GMC ở frame đen, reset track ở chuyển cảnh mạnh và đo riêng thời gian đọc/detection/tracking/vẽ/xuất file.

Xem [kết quả cải tiến và nghiệm thu sáu vấn đề](docs/IMPROVEMENTS.md), [sơ đồ hệ thống](docs/ARCHITECTURE.md) và [kế hoạch](PROJECT_PLAN.md). Metric chỉ có khi đối chiếu GT; số ID trong video không phải số người thật.

Lệnh kiểm tra:

```bash
python -m unittest discover -s tests -v
python scripts/scoring/verify_improvements.py
```

Lệnh `verify_improvements.py` kiểm tra lại bằng chứng nghiệm thu V2 đã lưu. Chạy toàn bộ MOT17 với V2 bằng `python scripts/inference/run_mot17.py --profile quality --output outputs/mot17_v2_new`; dùng tên mới để giữ kết quả V1.

Các phần bên dưới ghi lại **V1** và số liệu lịch sử. `outputs/mot17_full/`, `configs/models/yolo26s.json`, `configs/trackers/tracktrack.yaml` cùng quyết định `mot17_selected.json` vẫn giữ V1; CLI hiện đọc preset V2. Muốn tái hiện tracking V1, chỉ định `yolo26s_v1.json`, `tracktrack_v1.yaml` và `configs/runtime/legacy.json` qua ba cờ cấu hình, thay vì dùng preset mới.

## Chạy video với cấu hình V1

```bash
python scripts/inference/run_tracking.py --source /duong/dan/video.mp4 --model-config configs/models/yolo26s.json --tracker-config configs/trackers/tracktrack.yaml --output outputs/runs/my_video
```

Kết quả chính là `tracking.mp4`, `tracks.txt`, `tracks.jsonl`, `summary.json`. Thêm `--frames 100` để thử ngắn; dùng thư mục đầu ra mới hoặc rỗng. Xem [cài môi trường](#tạo-lại-môi-trường-cpu) và [báo cáo kết quả](docs/REPORT.md).

## Chạy toàn bộ MOT17

Các lệnh batch cho hai bộ dữ liệu:

```bash
python scripts/inference/run_mot17.py --benchmark MOT17 --profile quality --output outputs/mot_full_v2/mot17
python scripts/inference/run_mot17.py --benchmark MOT20 --profile quality --output outputs/mot_full_v2/mot20
```

Lệnh sau chạy và chấm riêng MOT17 train với ReID và nối quỹ đạo.

```bash
YOLO_AUTOINSTALL=false PYTHONPATH="$PWD/.runtime/reid-python" \
  python scripts/inference/run_mot17.py --profile reid --train-only --output outputs/mot17_train_reid_batch
```

Profile `reid` nối sau bù camera trên MOT17-05, 10, 11 và 13. MOT17-02, 04 và 09 dùng nối theo tọa độ ảnh. Neo vị trí trên 02 và 04 được áp dụng khi MOT TXT đầu vào khớp bản đã duyệt. Các sequence còn lại dùng đầu ra trực tiếp từ TrackTrack + ReID.

Dùng `--resume` nếu một lượt bị ngắt và đã có `manifest.json`. Chỉ chạy một tiến trình cho mỗi thư mục đầu ra. Các file ZIP và điểm train được xuất sau khi toàn bộ sequence của bộ đó hoàn tất.

Đã có kết quả đầy đủ tại `outputs/mot17_full/`, khoảng 1,6 GB: **14 video, 11.235 frame, 42 tệp MOT**. Tệp nén gồm [MOT TXT train](outputs/mot17_full/motchallenge/train.zip) và [MOT TXT test](outputs/mot17_full/motchallenge/test.zip). Kiểm chứng ở [mot17_full_validation.json](outputs/checks/mot17_full_validation.json).

Để tạo một lượt chạy mới:

```bash
python scripts/inference/run_mot17.py --output outputs/mot17_full_new
```

Lệnh xử lý toàn bộ 7 video train và 7 video test, tổng **11.235 frame khác nhau**. Ảnh DPM/FRCNN/SDP được đối chiếu SHA-256 ở mọi frame; mỗi video chạy một lần, sau đó xuất đủ **42 tệp MOT** theo tên của cả ba bản dataset. Các run đầy đủ đã có chỉ được dùng lại nếu khớp nguồn, cấu hình, phiên bản và số frame; run mới không giới hạn frame.

Đầu ra trong `outputs/mot17_full/`:

- `train/<sequence>/`, `test/<sequence>/`: video tracking, bản H.264 `preview.mp4`, MOT TXT, JSONL, ảnh đầu/cuối và summary.
- `configs/`: snapshot cấu hình; `manifest.json`: checksum ảnh, nguồn run dùng lại và tiến độ từng video.
- `motchallenge/train/`, `motchallenge/test/`: mỗi tập có 21 tệp TXT; `train.zip`, `test.zip` chứa các tệp tương ứng.
- `evaluation_train/`: metric trên toàn bộ 5.316 frame train có GT. Tập này gồm cả dữ liệu phát triển và giữ lại của dự án; test không có GT cục bộ để chấm.

Nếu lần chạy bị ngắt, thêm `--resume` vào lệnh chạy cùng `--output`; các video hoàn tất được giữ lại. Thư mục run mới phải rỗng. Tiến trình dùng cấu hình đã chốt, không chỉnh tham số theo kết quả toàn tập.

Lượt chạy đã hoàn tất gồm 11 video chạy mới và 3 video dùng lại đã xác minh (MOT17-09/11/13). Metric toàn bộ train 5.316 frame: **HOTA 45,307; IDF1 54,827; MOTA 44,669; 435 lần đổi ID**. Đây là điểm trên toàn bộ train, gồm cả dữ liệu phát triển và giữ lại; không so trực tiếp với điểm giữ lại 1.650 frame ở bước 5. Test chỉ xuất dự đoán vì không có GT cục bộ.

## Thử tham số — bước 5

```bash
conda activate mot
python scripts/experiments/tune_tracking.py --plan configs/experiments/mot17_step5.json --output outputs/experiments/my_study
```

Kế hoạch thử nghiệm chỉ nhận sequence thuộc tập phát triển; script từ chối MOT17-11/13. Mỗi biến thể thay đúng một tham số của cấu hình cha. `parent: best` nghĩa là cấu hình tốt nhất đã chấm trước đó, theo thứ tự HOTA → IDF1 → FPS. Hai thay đổi đầu đi theo chuỗi hạ ngưỡng tạo track rồi hạ ngưỡng high, để kiểm tra tương tác giữa hai ngưỡng. Kế hoạch và cấu hình thực chạy được lưu trước khi chấm từng biến thể.

Đầu ra: `comparison.json` ghi cấu hình cha, thay đổi và số liệu; `configs/`, `runs/`, `evaluations/` giữ cấu hình, video/track và kết quả đánh giá của từng biến thể. Dùng thư mục đầu ra mới hoặc rỗng. Đây là tìm kiếm có giới hạn trên các đoạn phát triển, không bảo đảm tối ưu toàn cục.

Sau khi chấm đủ các biến thể:

```bash
python scripts/experiments/tune_tracking.py --finalize-study outputs/experiments/my_study --output outputs/experiments/my_final
```

Lệnh này so baseline và ứng viên trên thêm 300 frame đầu mỗi video MOT17-02/05/10, gộp với hai đoạn ban đầu thành 1.625 frame phát triển để chọn cấu hình. Sau đó lưu snapshot/checksum trong `frozen/` và `selection.json` **trước khi** chạy toàn bộ MOT17-11/13. Kết quả giữ lại được báo cáo riêng trong `held_out/`, không dùng để chỉnh tiếp. Phạm vi và quy tắc nằm ở `configs/experiments/mot17_step5_confirmation.json`; lệnh không tự ghi đè cấu hình mặc định của mã nguồn.

Cũng có thể thử riêng mà không sửa cấu hình mặc định:

```bash
python scripts/inference/run_tracking.py --source data/MOT17/train/MOT17-09-FRCNN --model-config configs/models/yolo26s_baseline.json --tracker-config configs/trackers/tracktrack_baseline.yaml --output outputs/runs/custom_demo
```

`*_baseline` giữ nguyên cấu hình bước 4. `summary.json` ghi đường dẫn cấu hình và ReID thực tế: `model=auto` dùng đặc trưng của YOLO khi backend hỗ trợ; trường `frames_with_track_features` giúp kiểm tra nhánh ngoại hình có tạo đặc trưng. Nếu backend chuyển sang model khác thì tên model thực được ghi lại.

Cấu hình mặc định đã chốt ở bước 5: `imgsz=960`, `conf=0.01`; TrackTrack high/low/new = 0.3/0.1/0.3, match = 0.8, buffer = 60 frame; ReID tắt, GMC `sparseOptFlow`. Quyết định và checksum được lưu tại `configs/experiments/mot17_selected.json`. Trên 1.625 frame phát triển, HOTA **38,47 → 47,02**, IDF1 **42,70 → 57,30**, FPS CPU **12,55 → 6,89**. Ưu tiên chất lượng tổng thể; số dự đoán thừa và đổi ID vẫn tăng. Xem [báo cáo](docs/REPORT.md) để hiểu phạm vi và đánh đổi.

Đánh giá cuối trên toàn bộ 1.650 frame MOT17-11/13: HOTA **39,32 → 46,94**, IDF1 **43,89 → 55,92**, FPS CPU **12,34 → 6,73**. Kết quả ở `outputs/experiments/mot17_step5_final/held_out/`; có video của cả hai cấu hình và 5 clip lỗi mới cho cấu hình chốt. Hai video này thuộc `MOT17/train`, được giữ lại trong dự án; chưa phải kết quả benchmark test chính thức. Không chỉnh tiếp tham số sau khi xem điểm giữ lại.

## Đánh giá tracking — bước 4

```bash
conda activate mot
python scripts/inference/run_tracking.py --source data/MOT17/train/MOT17-09-FRCNN --output outputs/runs/eval_demo_09
python scripts/inference/run_tracking.py --source data/MOT17/train/MOT17-04-FRCNN --frames 200 --output outputs/runs/eval_demo_04
python scripts/scoring/evaluate.py --runs outputs/runs/eval_demo_09 outputs/runs/eval_demo_04 --output outputs/evaluations/eval_demo
```

Chọn các thư mục đầu ra mới hoặc rỗng. Chỉ chấm run tracking từ sequence MOT có GT; video tự quay chưa có nhãn không chấm được. Những run gộp chung phải dùng cùng model, cấu hình inference/tracker và phiên bản Ultralytics/PyTorch. Không đưa nhiều bản FRCNN/DPM/SDP của cùng video vào một phép so sánh.

Đầu ra đánh giá:

- `metrics.json`: HOTA, DetA, AssA, IDF1, MOTA, ID switches, TP/FN/FP, tốc độ và phạm vi dữ liệu.
- `trackeval/YOLO26_TrackTrack/`: summary và CSV chi tiết do TrackEval xuất; kết quả gộp lấy từ TrackEval, không lấy trung bình đơn giản giữa các video.
- `inputs/`: bản sao GT và kết quả MOT dùng để chấm. Nếu chỉ chạy 200 frame, GT và metadata sao chép cũng giới hạn ở 200 frame; dữ liệu gốc được giữ nguyên.
- `<sequence>_errors.json`: lỗi từng frame theo phép ghép CLEAR, tổng FN/FP/IDSW được đối chiếu với TrackEval.
- `error_clips/`: mặc định 5 clip và ảnh minh họa, kèm `index.json`. Đỏ: bỏ sót (FN), vàng: dự đoán thừa (FP), tím: đổi ID (IDSW); khung bình thường là GT xanh lá và dự đoán xanh lơ. Nhãn G/T dùng ID đã được TrackEval chuẩn hóa, không phải ID gốc trong `tracks.jsonl`. Thêm `--clips 0` để bỏ tạo clip.

Baseline đã chạy tại `outputs/evaluations/baseline_mot17/`: **HOTA 42,54; IDF1 46,13; 33 lần đổi ID; 12,43 FPS trên CPU**. Phạm vi là toàn bộ 525 frame MOT17-09 và 200 frame đầu MOT17-04, thuộc tập phát triển; đây là kết quả cục bộ, chưa phải điểm toàn bộ benchmark MOT17. Xem [báo cáo và 5 clip lỗi](docs/REPORT.md).

Đánh giá dùng `trackeval==1.3.0` trên [PyPI](https://pypi.org/project/trackeval/), phát hành từ [fork kovalp/TrackEval](https://github.com/kovalp/TrackEval), có nguồn gốc từ [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval). Bật tiền xử lý pedestrian của MOTChallenge (`DO_PREPROC=True`) để xử lý nhãn bỏ qua/distractor theo evaluator.

## Theo dõi người — bước 3

```bash
conda activate mot
python scripts/inference/run_tracking.py --source data/MOT17/train/MOT17-09-FRCNN --output outputs/runs/tracking_demo
```

`--source` nhận video cục bộ hoặc thư mục sequence MOT. Bỏ tham số này để dùng MOT17-09-FRCNN. Mặc định xử lý hết nguồn; thêm `--frames 100` để chạy ngắn. Chọn thư mục `--output` mới hoặc rỗng.

Đầu ra:

- `tracking.mp4`: khung bao, ID và quỹ đạo 30 khung hình gần nhất.
- `tracks.jsonl`: mỗi frame có danh sách track gồm `id`, `xyxy`, `confidence`, `class_id`; frame chưa có ID vẫn được ghi với danh sách rỗng.
- `tracks.txt`: định dạng MOTChallenge 10 cột `frame,id,left,top,width,height,confidence,-1,-1,-1`. Frame và tọa độ MOT bắt đầu từ 1; JSONL vẫn dùng `xyxy` gốc 0. Khi xuất, `left=x1+1`, `top=y1+1`, kích thước giữ nguyên.
- `frame_000001.jpg`, `frame_<cuối>.jpg`: ảnh đầu/cuối đoạn.
- `summary.json`: cấu hình model/tracker, checksum, số frame, quan sát tracking, FPS, độ trễ median/p95 và phần cứng.

Phép đo tốc độ chạy sau một lần warm-up detector: tính đọc frame, khởi tạo/chạy tracker, vẽ, ghi JSONL/MOT và hoàn tất video; không tính nạp model hay warm-up. FPS gộp bằng tổng frame chia tổng thời gian trên cùng phần cứng. Đây là một lần đo cục bộ; không bao gồm thời gian chấm TrackEval/tạo clip lỗi. Run cũ thiếu `timing_protocol` sẽ có FPS đánh giá là `null`; muốn so tốc độ cần chạy lại.

Lệnh này tự chạy detection và tracking trực tiếp từ nguồn ảnh/video. Tracker cập nhật ở mọi frame, kể cả khi không phát hiện người; model, tracker và lịch sử quỹ đạo được tạo mới cho mỗi nguồn. Chỉ vẽ track được backend trả về với ID ở frame hiện tại. Quỹ đạo lấy điểm giữa cạnh dưới khung bao theo pixel ảnh, không nối đường qua khoảng mất quan sát; lịch sử cũ được dọn sau 30 frame.

Cấu hình inference lấy từ `configs/models/yolo26s.json`, tham số tracking lấy từ `configs/trackers/tracktrack.yaml`. Ngưỡng `display_confidence` của bước 2 không lọc video tracking; việc gán/giữ ID do TrackTrack quyết định. Box xuất ra được giới hạn trong ảnh gốc mà không sửa trạng thái bên trong tracker. ID chỉ có ý nghĩa trong từng lần chạy, không phải số người thật.

Kết quả bước 3 nằm ở `outputs/runs/track_mot17_09/` (525 frame) và `outputs/runs/track_mot17_04/` (200 frame). Bước 4 chạy lại cùng cấu hình tại `baseline_mot17_09/` và `baseline_mot17_04/` để xuất MOT và đo tốc độ sau warm-up. Tracker vẫn giữ nguyên: ReID tắt, GMC `sparseOptFlow`.

Giới hạn đã thấy: clip thử 20 frame có một frame đen đột ngột vẫn cập nhật tracker đủ 20 lần, nhưng sau đó các đối tượng nhận ID mới. Chẩn đoán ghi nhận GMC ước lượng chuyển động bất thường tại frame đen. Test backend với detection đơn giản và GMC tắt giữ được ID qua một frame trống; kết quả đó không bảo đảm khôi phục ID trong mọi video. Bước 5 chạy lại cấu hình chốt trên cùng clip vẫn mất ID sau frame đen; kết quả lưu ở `outputs/checks/tuning_validation.json`, video tại `outputs/runs/step5_gap_check/`. Chẩn đoán ban đầu được giữ ở `outputs/checks/tracking_validation.json`.

## Phát hiện người — bước 2

```bash
conda activate mot
python scripts/inference/run_detection.py --source data/MOT17/train/MOT17-09-FRCNN --output outputs/runs/detection_demo
```

`--source` nhận thư mục sequence có `seqinfo.ini` hoặc tệp video cục bộ. Bỏ `--source` để dùng MOT17-09-FRCNN. Mặc định chạy hết nguồn; thêm `--frames 100` để chạy ngắn. Mỗi lần thử dùng một thư mục `--output` mới hoặc rỗng, tránh ghi đè kết quả cũ.

Ví dụ video:

```bash
python scripts/inference/run_detection.py --source path/to/input.mp4 --frames 100 --output outputs/runs/video_demo
```

Đầu ra trong thư mục đã chọn:

- `detections.mp4`: video chỉ có khung bao và confidence, chưa gán ID.
- `detections.jsonl`: mỗi dòng là một frame và danh sách detection, kể cả danh sách rỗng.
- `frame_000001.jpg`, `frame_<cuối>.jpg`: ảnh xem nhanh đầu/cuối đoạn.
- `summary.json`: cấu hình thực chạy, số detection, số frame chạm `max_det`, checksum và thời gian xử lý.

Cấu hình inference hiện nằm trong `configs/models/yolo26s.json`: `conf=0.01`, `imgsz=960`, `iou=0.7`, `max_det=1000`, `nms=null` dùng nhánh NMS. Video chỉ hiển thị detection từ `display_confidence=0.25`; tệp JSONL giữ cả các box từ 0.01. Kết quả bước 2 đã lưu trước đó dùng cấu hình baseline `conf=0.05`, `imgsz=640`. Số detection là tổng quan sát theo frame, không phải số người duy nhất.

Tọa độ JSONL là `xyxy` theo pixel ảnh gốc, gốc tọa độ `(0, 0)`; frame bắt đầu từ 1. Video kích thước lẻ được thêm tối đa một hàng/cột đen để encoder MP4 ghi đủ ảnh; tọa độ detection vẫn theo ảnh gốc. Tệp này là cache detection, chưa phải kết quả MOT có ID.

Đã chạy kiểm tra trên toàn bộ 525 frame MOT17-09 (`outputs/runs/detect_mot17_09/`) và 200 frame MOT17-04 (`outputs/runs/detect_mot17_04/`). Video đọc lại đủ frame; hai đoạn chưa chạm giới hạn `max_det=1000`. Preview vẫn có box ở vùng ma-nơ-canh/tủ kính; một số người xa/che khuất không hiện box ở ngưỡng vẽ 0.25. Cần xem cả cache 0.05 và đối chiếu nhãn trước khi kết luận chất lượng detection.

## Chạy trong môi trường đã cài

Từ thư mục dự án:

```bash
conda activate mot
python scripts/setup/check_setup.py
python scripts/setup/smoke_test.py --frames 100
```

Script chọn GPU nếu PyTorch nhận được CUDA, còn lại dùng CPU. Môi trường được thiết lập trong phiên này là bản CPU; muốn dùng CUDA sau này cần cài PyTorch phù hợp với GPU/driver.

Kết quả:

- `outputs/checks/setup_report.json`: phiên bản thư viện, thiết bị và kiểm tra dữ liệu.
- `outputs/runs/smoke_test/preview.mp4`: video 100 khung hình có khung bao và ID.
- `outputs/runs/smoke_test/frame_000100.jpg`: ảnh kết quả cuối đoạn thử.
- `outputs/runs/smoke_test/summary.json`: số khung hình đã chạy, backend tracker và thông tin trọng số.

Đây là kiểm tra chạy được pipeline, chưa chấm HOTA/IDF1. Số ID được tạo không phải số người thật và không đo chất lượng tracking. Chạy lại sẽ ghi đè kết quả smoke test cũ.

## Dữ liệu

Giữ dữ liệu đã tải tại `data/MOT17/train/<tên sequence>/`, gồm `seqinfo.ini`, `img1/` và `gt/gt.txt`. Thư mục `MOT17Labels` được giữ nguyên; script dùng nhãn đã có trong `MOT17`.

Cấu hình chia tập nằm ở `configs/datasets/mot17.json`:

- Phát triển/chỉnh tham số: MOT17-02, 04, 05, 09, 10.
- Giữ lại cho đánh giá cuối: MOT17-11, 13.
- Chạy thử: 100 khung hình đầu của MOT17-09-FRCNN.

Chỉ chọn bản FRCNN của mỗi video để tránh lặp dữ liệu. Hậu tố này là tên bản dataset; chương trình vẫn chạy detector YOLO26s, không lấy detection từ Faster R-CNN. Không chuyển các bản DPM/SDP của cùng video sang tập đánh giá.

## Trọng số và tracker

- Detector: `weights/yolo26s.pt`, bản Detect pretrained do Ultralytics phát hành. Nguồn, SHA-256 và thông tin training đọc từ checkpoint được lưu tại `configs/models/yolo26s.json`.
- Lệnh detection/tracking đọc cấu hình model này và kiểm tra checksum trước khi chạy. Lệnh smoke test bước 1 vẫn giữ cấu hình chạy thử riêng trong `smoke.py`.
- Tracker: bản tích hợp Ultralytics, cấu hình gốc được lưu ở `configs/trackers/tracktrack.yaml`; ReID đang tắt, GMC dùng `sparseOptFlow`.
- Nguồn Ultralytics mô tả trọng số detection pretrained trên COCO. Chưa xác minh độc lập việc trùng ảnh giữa dữ liệu huấn luyện và MOT17; chia tập cục bộ không tự chứng minh checkpoint chưa thấy dữ liệu.

Tải lại trọng số nếu cần:

```bash
mkdir -p weights
curl --fail --location --output weights/yolo26s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

## Tạo lại môi trường CPU

`requirements.txt` khóa các phiên bản đã cài cho Python 3.11 trên Linux x86_64, gồm PyTorch CPU.

```bash
conda create -n mot python=3.11 pip
conda activate mot
python -m pip install -r requirements.txt
```

Bỏ qua bước tạo env nếu `mot` đã tồn tại. Cài đặt/tải trọng số cần mạng; các script kiểm tra chạy cục bộ sau khi đã có đủ thư viện và trọng số.

## Cấu trúc dự án

- `src/mot/tracking/pipeline.py`: tracking đầy đủ, giữ trạng thái, xuất video/JSONL/MOT và đo tốc độ.
- `src/mot/tracking/mot_io.py`: chuyển tọa độ và xuất kết quả MOTChallenge.
- `src/mot/evaluation/metrics.py`, `src/mot/evaluation/diagnostics.py`: chấm TrackEval, đối chiếu lỗi từng frame và tạo clip minh họa.
- `src/mot/tracking/visualization.py`: vẽ ID và quỹ đạo ngắn, quản lý lịch sử hiển thị.
- `src/mot/tracking/smoke.py` chạy kiểm tra ngắn qua lệnh `scripts/setup/smoke_test.py`.
- `src/mot/tracking/detection.py`, `src/mot/tracking/sources.py`: detection riêng và đọc nguồn video/chuỗi ảnh.
- `scripts/setup/` kiểm tra môi trường, `scripts/inference/` chạy video và MOT17, `scripts/scoring/` chấm điểm, `scripts/experiments/` chạy thử nghiệm và `scripts/web_tools/` khởi động và kiểm tra camera trực tiếp.
- `src/mot/app/` chứa xử lý RTSP, API và giao diện xem camera.
- `scripts/experiments/tune_tracking.py`, `configs/experiments/`: thử từng thay đổi trên tập phát triển và lưu số liệu so sánh.
- `scripts/inference/run_mot17.py`: chạy đầy đủ train/test, kiểm tra ảnh trùng, dùng lại run phù hợp và xuất tệp theo cả ba hậu tố.
- `configs/`: cấu hình dataset, thông tin model và cấu hình tracker theo từng thư mục riêng.
- `data/`, `weights/`: dữ liệu và trọng số đã tải.
- `outputs/checks/`, `outputs/runs/`, `outputs/evaluations/`, `outputs/experiments/`: kiểm tra, dự đoán, đánh giá và so sánh cấu hình.
- `docs/`: sơ đồ toàn hệ thống, cấu trúc đích và nhiệm vụ của từng module.

Xem [sơ đồ và định nghĩa cấu trúc](docs/ARCHITECTURE.md).

## Kiểm tra mã nguồn

```bash
python -m unittest discover -s tests -v
```

Các test dùng ảnh/video tạm và giả lập detection; các kiểm tra vòng đời tracking dùng backend TrackTrack thật, các kiểm tra metric dùng TrackEval thật. Bao gồm frame trống, reset nguồn, quỹ đạo, tọa độ MOT, tracking hoàn hảo, đổi ID, không có dự đoán, distractor, GT của đoạn ngắn và metric gộp. Bước 5 kiểm tra chuyển cấu hình, chặn tập giữ lại và đóng băng trước đánh giá cuối. Công cụ chạy toàn bộ MOT17 kiểm tra ảnh trùng ở mọi frame, dữ liệu thiếu và điều kiện dùng lại run. Inference thực tế được kiểm tra bằng `verify_camera.py` và các run thực nghiệm.

Tài liệu: [kế hoạch dự án](PROJECT_PLAN.md), [YOLO26](https://docs.ultralytics.com/models/yolo26/), [TrackTrack trong Ultralytics](https://docs.ultralytics.com/modes/track/).
