# Triển khai camera IP nội bộ

Bản đầu chạy một camera cố định qua RTSP trên mạng LAN. Dịch vụ dùng YOLO26s OpenVINO FP32 ở `imgsz=1280`, TrackTrack và ReID, với mục tiêu tracking 4 FPS. Web hiển thị khung bao và ID từ ảnh mới nhất trong RAM. Journal lưu trạng thái kết nối và lỗi dịch vụ.

## Chạy thử trên máy hiện tại

Đặt URL RTSP vào một tệp quyền `0600` trong `.runtime/camera/`. Khi thử từ env `mot` với các runtime đã dùng để benchmark, chạy từ thư mục dự án.

```bash
conda activate mot
PYTHONPATH="$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/web_tools/serve_camera.py \
  --rtsp-url-file .runtime/camera/rtsp-url
```

Mở [trang xem camera](http://127.0.0.1:8003). Máy chủ gửi ảnh JPEG có box/ID trên `/api/camera/stream` với mục tiêu 30 FPS và chiều rộng không quá 1280 px. Optical flow dịch chuyển box theo từng ảnh. YOLO + TrackTrack + ReID cấp hoặc sửa ID với mục tiêu 4 FPS. Nếu kết quả tracking gần nhất cũ hơn 800 ms, ảnh hiển thị không có box. Trình duyệt đệm khoảng 100 ms lúc bắt đầu để giữ nhịp đều hơn.

Giao diện hiển thị FPS ảnh vẽ trên trình duyệt, FPS optical flow tại máy chủ và FPS tracking. `GET /api/camera/status` cũng trả về số frame đã xử lý, số frame được thay thế, tuổi ảnh và thời gian trung bình ở từng bước. `GET /api/camera/live.jpg` trả ảnh hiển thị mới nhất và `GET /api/camera/frame.jpg` trả ảnh tại frame YOLO vừa tracking. Cả hai dùng `ETag` hoặc trả `503` khi hết tín hiệu. Mỗi phiên ID mới có định danh riêng sau khi kết nối lại, đổi độ phân giải hoặc chuyển cảnh.

Nếu chưa có camera, chuẩn bị MediaMTX và FFmpeg. Lệnh dưới chạy relay, publisher và viewer trên loopback, phát video MOT17-04 ở 1080p 30 FPS, ngắt và nối lại publisher, rồi đo FPS và RAM trong 30 phút. Thư mục kết quả phải là thư mục mới.

```bash
PYTHONPATH="$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/web_tools/verify_camera.py \
  --mediamtx /path/to/mediamtx --seconds 1800 \
  --output outputs/camera_rtsp_check_4fps
```

[Phát RTSP với FFmpeg và MediaMTX](https://mediamtx.org/docs/publish/ffmpeg)

### Xem optical flow bằng luồng RTSP mô phỏng

Phát lại clip mô phỏng rồi mở `http://127.0.0.1:8003` khi terminal báo viewer đang chạy:

```bash
PYTHONPATH="$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/web_tools/verify_camera.py \
  --mediamtx .runtime/camera-test/mediamtx \
  --fixture outputs/camera_demo_4fps/fixture.mp4 \
  --seconds 180 --output outputs/camera_optical_flow_check
```

### Đo FP32, FP16 và INT8 ở imgsz 1280

`scripts/experiments/benchmark_camera_precision.py --output <thư-mục-mới>` dùng cùng tracker và cùng bộ ảnh MOT17-04 lấy mẫu 4 FPS. Bài đo offline không giới hạn thông lượng; số FPS đo được bao gồm phát hiện, ghép track, đọc ảnh và xuất kết quả. Bản INT8 được hiệu chuẩn bằng NNCF trên MOT17-02 và MOT17-09. Cài `nncf==3.4.0` trong môi trường benchmark có sẵn OpenVINO và ReID.

Trên i5-14400, FP32, FP16 và INT8 lần lượt đạt 4,52, 4,45 và 5,27 FPS offline. INT8 giảm HOTA 12,36 điểm và IDF1 15,58 điểm so với FP32 trên MOT17-04. Dịch vụ vẫn dùng FP32. Xem `outputs/camera_precision_comparison/comparison.json` để biết các số đo chi tiết.

### Thử WebRTC trên máy không có camera

Chạy từ thư mục dự án với MediaMTX đã đặt tại `.runtime/camera-test/mediamtx`. Có thể bỏ `--fixture` để mã hóa lại MOT17-04 từ ảnh gốc. Mỗi lần chạy chọn `--output` mới.

```bash
PYTHONPATH="$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/web_tools/verify_camera.py \
  --mediamtx .runtime/camera-test/mediamtx \
  --fixture outputs/camera_smooth_4fps_resized_20261123/fixture.mp4 \
  --webrtc --seconds 120 --output outputs/camera_webrtc_demo
```

Mở `http://127.0.0.1:8889/camera` để xem **video nguồn không có box** qua WebRTC ở 30 FPS. Mở `http://127.0.0.1:8003` để xem ảnh có box/ID do optical flow cập nhật. Bài thử chỉ mở cổng trên loopback và tự dừng. Trang WebRTC của MediaMTX dành cho thử nghiệm tại máy chạy dịch vụ.

Trong lần thử trên MOT17-04, Chrome trình bày 29,2 FPS WebRTC trong 10 giây, còn tracking đạt 3,88 FPS trong cửa sổ một phút. Nén trọng số OpenVINO sang FP16 làm tệp `.bin` giảm từ 38,0 MB xuống 19,0 MB; trên 200 frame đầu MOT17-04, FP32 camera cố định đạt 4,33 FPS và FP16 đạt 4,28 FPS. Bỏ GMC trên các sequence cố định lấy mẫu 4 FPS làm HOTA/IDF1 validation giảm 0,39/0,22 điểm và FPS validation giảm 0,10; trên nhóm development FPS tăng 0,14, nên chưa có mức tăng ổn định. Vì vậy profile triển khai vẫn dùng FP32 và GMC; các biến thể thử nghiệm nằm ở `configs/models/yolo26s_1280_openvino_fp16.json` và `configs/trackers/tracktrack_camera_fixed_4fps.yaml`. Xem số đo tại `outputs/camera_webrtc_trial/comparison.json`.

### Thử tracking giữa các frame detection

`scripts/experiments/benchmark_hybrid_camera.py` chạy YOLO26s + TrackTrack + ReID mỗi 6 hoặc 8 frame, rồi dùng optical flow để cập nhật vị trí của **cùng ID** trên từng frame xen giữa. TrackTrack vẫn là nơi cấp ID; optical flow không tự nhận dạng lại người sau khi mất dấu. Ví dụ chạy trên toàn bộ MOT17-04:

```bash
PYTHONPATH="$PWD/src:$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/experiments/benchmark_hybrid_camera.py \
  --source data/MOT17/train/MOT17-04-FRCNN --frames 1050 --interval 8 \
  --reference outputs/mot17_train_openvino_fp32/online/train/MOT17-04-FRCNN \
  --output outputs/camera_hybrid_flow_full_20261123
```

Kết quả trên CPU, chấm TrackEval với ground truth toàn video:

- MOT17-04, mỗi 8 frame: **27,28 FPS**, HOTA **52,36**, IDF1 **63,41**; giữ nguyên box đạt HOTA 50,30, IDF1 62,11. Bản tham chiếu chạy detector mọi frame đạt HOTA 53,03, IDF1 63,95. Có bản xem thử 10 giây ở `outputs/camera_hybrid_flow_full_20261123/preview_10s.mp4`.
- MOT17-02, mỗi 6 frame: **28,20 FPS**, HOTA **36,64**, IDF1 **46,52**; bản tham chiếu HOTA 36,88, IDF1 44,56.
- MOT17-09, mỗi 6 frame: **27,61 FPS**, HOTA **50,07**, IDF1 **64,28**; bản tham chiếu HOTA 48,29, IDF1 59,25. Bản chạy mỗi 8 frame nhanh hơn nhưng giảm chất lượng đáng kể trên 02 và 09.

Các FPS này là **thông lượng xử lý tuần tự** gồm đọc ảnh, detection thưa, optical flow và ghi MOT; chưa gồm WebRTC, vẽ box hoặc độ trễ phát trực tiếp. Bản tham chiếu dùng cùng checkpoint và tham số YOLO nhưng cấu hình TrackTrack cho video đầy đủ khác cấu hình camera thưa; so sánh với cách giữ box dùng đúng cùng cấu hình và cùng các frame detection. Lần detection vẫn gây ngắt quãng khoảng 170–200 ms, nên chưa thể kết luận trang camera hiển thị box mượt 25–30 FPS. Bản thử chưa thay profile dịch vụ camera; cần tách detection sang worker và đo FPS/độ trễ đồng thời với WebRTC trước khi đưa vào vận hành. Kết quả đầy đủ ở `outputs/camera_hybrid_flow_full_20261123/report.json`, `outputs/camera_hybrid_flow_mot17_02_i6_20261123/report.json` và `outputs/camera_hybrid_flow_mot17_09_i6_20261123/report.json`.

### Đo đồng thời WebRTC và tracking trên luồng RTSP mô phỏng

`scripts/experiments/verify_hybrid_live.py` phát lại MOT17-04 1080p30 qua MediaMTX, cho Chrome xem WebRTC và chạy tracker CPU ở hai worker: detection định kỳ, optical flow từng frame. Khi detection xong, vị trí được chiếu từ frame đã phát hiện tới frame mới nhất. Chạy với thư mục kết quả mới:

```bash
PYTHONPATH="$PWD/src:$PWD/.runtime/openvino-python:$PWD/.runtime/reid-python" \
  python scripts/experiments/verify_hybrid_live.py \
  --interval 8 --seconds 20 --output outputs/camera_hybrid_live_check
```

Trong cửa sổ 30 giây, Chrome hiển thị **28,03 FPS**, optical flow cập nhật quỹ đạo **30,17 FPS** trong 10 giây cuối, còn YOLO + TrackTrack + ReID đạt **3,70 FPS**. Không rơi frame ở hàng đợi optical flow. Cửa sổ 20 giây đo thêm độ đều: WebRTC **26,95 FPS**, khoảng cách frame p95 **65 ms**; tracking **30,04 FPS**, khoảng cách cập nhật p95 **92 ms**. Độ trễ detection p95 **496 ms**; 11 frame chờ detection bị thay bằng frame mới hơn. Vì vậy **thông lượng trung bình qua 25 FPS đã đạt, nhưng nhịp hiển thị và nhịp tracking chưa đều ở 25 FPS**. Trình duyệt hiện xem video nguồn không có box; quỹ đạo/ID được đo trong worker CPU, chưa được ghép lên video WebRTC. Cần giảm các khoảng dừng và đo trực tiếp overlay trên trình duyệt trước khi thay dịch vụ camera hiện tại. Xem `outputs/camera_hybrid_live_i8_20261125/report.json` và `outputs/camera_hybrid_live_jitter_20261125/report.json`.

Lệnh trên cũng ghi `tracking.mp4` trong thư mục `--output`: video 960×540, 30 FPS, mỗi frame có box theo optical flow và ID do TrackTrack/ReID cấp. Xem bằng `ffplay outputs/camera_hybrid_live_check/tracking.mp4` sau khi lệnh kết thúc. Bản đã kiểm tra là `outputs/camera_hybrid_live_video_20261125/tracking.mp4`, gồm 665 frame trong 22,17 giây. Khi vừa ghi video vừa đo WebRTC, tốc độ hiển thị nguồn đạt **26,05 FPS**, cập nhật tracking **29,90 FPS** nhưng khoảng cách cập nhật p95 **103 ms**; chi tiết tại `outputs/camera_hybrid_live_video_20261125/report.json`.

Video mới dùng lại `TrackVisualizer` để chọn màu theo ID và vẽ vệt quỹ đạo như pipeline trước đây; sau khi đo FPS, file được chuyển sang H.264. Xem bản đã tạo bằng:

```bash
ffplay outputs/camera_hybrid_colored_web_20261125/tracking.mp4
```

Muốn tạo bản mới, chạy lại `verify_hybrid_live.py` với thư mục `--output` mới rồi mở `tracking.mp4` trong thư mục đó. Bản màu đã kiểm tra dài 14,73 giây, 442 frame ở 30 FPS; khi đồng thời ghi video và đo WebRTC, tốc độ tracking trung bình đạt 29,85 FPS, còn nhịp cập nhật p95 vẫn là 124 ms. Đây là video ghi lại, chưa phải overlay WebRTC phát trực tiếp.

## Chuẩn bị bản triển khai

1. Đặt một bản phát hành có phiên bản dưới `/opt/mot/releases/`. Bản phát hành gồm `src/`, `scripts/`, `configs/`, `deploy/`, `requirements.txt`, thư mục `weights/yolo26s_openvino_model/` đầy đủ và `weights/yolo26n-reid.onnx`. Dùng Python 3.11 cài dưới `/usr` hoặc `/opt` để tài khoản dịch vụ đọc được runtime. Tạo `.venv` trong bản phát hành và cài `deploy/camera/requirements.txt`. Trỏ `/opt/mot/current` vào phiên bản đã kiểm tra. Trình nạp kiểm tra checksum của OpenVINO và ReID khi khởi động.
2. Tạo tài khoản hệ thống `mot` cùng group `mot`. Đặt URL của tài khoản chỉ đọc trên camera vào `/etc/mot/rtsp-url` với quyền `0600`, chủ sở hữu root. Cài `mot-camera.service` vào `/etc/systemd/system/`. Chạy `systemctl daemon-reload` rồi `systemctl start mot-camera` để nghiệm thu. Xem log bằng `journalctl -u mot-camera`.
3. Trỏ `mot.home.arpa` về IP máy chủ trên DNS cục bộ hoặc máy xem. Cài Caddy và dùng [Caddyfile](Caddyfile). Đặt dòng `viewer <hash>` trong `/etc/caddy/mot-camera-users.caddy`, tạo hash bằng `caddy hash-password` và giới hạn quyền đọc cho tài khoản dịch vụ Caddy. Cho máy xem tin cậy CA nội bộ của Caddy, rồi mở `https://mot.home.arpa`. Giới hạn cổng HTTPS cho mạng LAN. Kiểm tra truy cập chưa đăng nhập trả `401` và sau đăng nhập cả trang lẫn API trả kết quả. [Caddy HTTPS nội bộ](https://caddyserver.com/docs/automatic-https#local-https) · [Xác thực Caddy](https://caddyserver.com/docs/caddyfile/directives/basic_auth)

## Nghiệm thu và quay lui

- Chấm lại sau mỗi lần thay model hoặc ngưỡng tracker bằng `scripts/experiments/tune_camera_tracking.py`. Dùng kết quả OpenVINO online toàn video làm `--reference` và thư mục mới làm `--output`. Chọn cấu hình trên MOT17-02/09, chấm MOT17-04 với cùng frame và ground truth cho mỗi mức FPS. Ngưỡng chất lượng trên 04 là giảm tối đa 2 điểm HOTA và 3 điểm IDF1.
- Trên camera thật, đo ít nhất 30 phút gồm lúc đông người và một lần mất mạng. Yêu cầu ít nhất 3,8 FPS trong 95% cửa sổ một phút có tín hiệu, RAM đỉnh tối đa 1,5 GiB và khôi phục trong 15 giây sau khi camera phát lại. Đo riêng p95 từ cảnh thật đến trình duyệt bằng đồng hồ mili giây đặt trong khung hình, mục tiêu không quá 1,2 giây. Kiểm tra chạy tiếp 24 giờ, trạng thái kết nối và mức sử dụng RAM ổn định, rồi dùng `systemctl enable mot-camera`.
- Để quay về bản phát hành trước, trỏ `/opt/mot/current` về bản cũ, khôi phục unit systemd tương ứng và khởi động lại dịch vụ.
