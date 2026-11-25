# Dự án: Theo dõi người đi bộ bằng YOLO26 + TrackTrack

## Cải tiến V2 — sáu vấn đề cần nghiệm thu

Giữ nguyên YOLO26 **Detect** + TrackTrack. CLI đọc `configs/profiles/{quality,fast}.json`. Những trạng thái bước 1–6 bên dưới là lịch sử V1, không thay thế phép nghiệm thu V2.

1. **Bỏ sót:** phân tích cache detection so với GT theo kích thước người; tăng độ phân giải và chọn lại tham số trên đúng 1.625 frame phát triển. Đạt khi FN giảm ít nhất 3%, HOTA không giảm và giải trình đủ tổng FN.
2. **Dự đoán thừa:** phối hợp ngưỡng tạo track và số frame xác nhận. Đạt khi FP giảm ít nhất 5%, recall không giảm quá một điểm phần trăm.
3. **Giữ ID:** bảo vệ GMC khi frame đen/ước lượng sai, giữ bộ đếm ID khi reset do chuyển cảnh mạnh. Đạt khi IDSW không tăng, IDF1 không giảm và regression frame đen/chuyển cảnh qua kiểm tra.
4. **Tốc độ CPU:** tách preset nhanh/chất lượng, đặt đúng số luồng và đo từng công đoạn. Đạt khi bản nhanh có ít nhất 15 FPS, nhanh hơn V1 ít nhất 1,5 lần, HOTA giảm không quá 8 điểm so với chất lượng. Báo riêng mục tiêu 30 FPS.
5. **Tổng quát hóa:** đóng băng cấu hình/checksum trước khi chấm MOT20-01 toàn bộ 429 frame và MOT20-03 frame 1–300. Đạt khi HOTA V2 không thấp hơn baseline quá 0,5 điểm trên cùng cửa sổ. Không chỉnh theo MOT20; MOT17-11/13 đã được xem ở V1 nên không gọi là dữ liệu mới.
6. **Ứng dụng (nghiệm thu V2 trước đây):** upload video thật, xử lý qua một worker và hỗ trợ tiến độ, preview, tải tệp, hủy và restart. Tiêu chí này áp dụng cho lần nghiệm thu lịch sử; hiện video đã ghi dùng CLI và giao diện web phục vụ [camera IP](../deploy/camera/README.md).

Ngưỡng nằm tại [six_issues_v2.json](configs/experiments/six_issues_v2.json); kết luận và giới hạn tại [IMPROVEMENTS.md](docs/IMPROVEMENTS.md). Đạt ngưỡng trên cửa sổ kiểm tra không có nghĩa đã hết lỗi hoặc đạt SOTA. Các đầu ra toàn bộ MOT17 trước đây vẫn là V1.

**Đã nghiệm thu V2: 6/6 ngưỡng đạt, 63 test đạt.** Trên MOT17 phát triển, FN giảm 3,85%, FP giảm 7,09%, IDSW 148 → 145. CPU bản nhanh đạt 18,875 FPS trong benchmark. MOT20 729 frame: HOTA quality 42,193 → 49,132, nhưng FP/IDSW tăng; xem giới hạn trong báo cáo. [Bằng chứng](outputs/improvements/verification.json).

## 1. Mục tiêu

Xây dựng chương trình nhận video từ một camera, phát hiện người đi bộ và giữ ID của từng người qua các khung hình, kể cả khi họ đi sát nhau hoặc bị che khuất ngắn.

Sản phẩm cuối: video có khung bao, ID và quỹ đạo; tệp kết quả tracking; báo cáo chất lượng và tốc độ.

Thời gian dự kiến: 6 tuần, khoảng 8–12 giờ/tuần. Bắt đầu bằng mô hình đã huấn luyện sẵn.

## 2. Cách hoạt động

Luồng chính: video/chuỗi ảnh → đọc từng khung hình → YOLO26 phát hiện người → TrackTrack giữ ID → vẽ khung bao, ID, quỹ đạo → xuất video và tệp tracking.

Luồng đánh giá: tệp tracking + nhãn chuẩn → TrackEval → HOTA, IDF1, số lần đổi ID; tốc độ được đo trong lần chạy pipeline.

Xem [sơ đồ toàn hệ thống và định nghĩa folder/file](docs/ARCHITECTURE.md), gồm phần đã triển khai và phần sẽ làm ở các bước tiếp theo.

- **YOLO26s:** xác định vị trí người trong từng khung hình.
- **TrackTrack:** nối các lần phát hiện của cùng một người qua thời gian.
- **Python, PyTorch, Ultralytics, OpenCV:** chạy mô hình và xử lý video.
- **TrackEval:** đánh giá kết quả khi có nhãn đúng để đối chiếu.

ID là mã theo dõi trong video, không phải danh tính thật của người.

## 3. Kế hoạch từng bước

### Bước 1 — Chuẩn bị môi trường và dữ liệu · Tuần 1

1. Kiểm tra Python, GPU và cài các thư viện cần thiết.
2. Chọn phiên bản Ultralytics có hỗ trợ YOLO26 và TrackTrack; lưu phiên bản chạy thành công.
3. Lấy một video MOT17 ngắn để thử, sau đó chọn các video MOT20 có cảnh đông người.
4. Tách video dùng chỉnh tham số và video dùng đánh giá cuối; không dùng cùng video cho cả hai.
5. Kiểm tra nguồn huấn luyện của trọng số để biết dữ liệu đánh giá có từng được mô hình học hay chưa.

**Hoàn thành khi:** đọc được video và chạy mô hình trên khoảng 100 khung hình.

**Trạng thái 17/11/2026:** đã cài môi trường `mot` (Python 3.11, Ultralytics 8.4.215, PyTorch CPU), kiểm tra 7 video MOT17 và chia tập phát triển/đánh giá. YOLO26s Detect + TrackTrack chạy thành công 100 khung hình của MOT17-09-FRCNN. Xem lệnh chạy trong [README](README.md). MOT20 sẽ bổ sung sau; độ chính xác tracking chưa được chấm.

### Bước 2 — Phát hiện người bằng YOLO26 · Tuần 2

1. Chạy YOLO26s, chỉ giữ lớp người.
2. Vẽ khung bao và kiểm tra trên video: người nhỏ, đứng sát nhau, thiếu sáng hoặc bị che khuất.
3. Kiểm tra tọa độ khung bao và giới hạn số người được xuất ra mỗi khung hình.
4. Giữ đủ các phát hiện có độ tin cậy thấp để TrackTrack còn cơ hội nối lại đối tượng.

**Hoàn thành khi:** có video phát hiện người với khung bao đúng vị trí.

**Trạng thái 18/11/2026:** đã có lệnh `scripts/inference/run_detection.py` cho video/chuỗi ảnh, xuất video và cache detection theo tọa độ ảnh gốc. Đã chạy toàn bộ 525 frame MOT17-09 và 200 frame MOT17-04; giữ confidence từ 0.05, hiển thị từ 0.25. Các kiểm tra nguồn ảnh/video, frame trống và tọa độ đã qua; độ chính xác detection chưa được chấm.

### Bước 3 — Gắn TrackTrack để giữ ID · Tuần 3

1. Chạy YOLO26 kết hợp TrackTrack, chỉ định rõ cấu hình `configs/trackers/tracktrack.yaml`.
2. Giữ trạng thái tracker xuyên suốt video; cập nhật cả khung hình không phát hiện được người.
3. Vẽ ID và quỹ đạo ngắn; khởi tạo lại tracker khi chuyển video.
4. Kiểm tra người cắt nhau, bị che khuất, đi vào và ra khỏi cảnh.

**Hoàn thành khi:** chạy hết một video, xuất được khung bao và ID mà không bị khởi tạo lại ở mỗi khung hình.

**Trạng thái 18/11/2026:** đã có `scripts/inference/run_tracking.py` cho video/chuỗi ảnh, xuất khung bao, ID, quỹ đạo 30 frame và `tracks.jsonl`. Đã chạy 525 frame MOT17-09, 200 frame MOT17-04; kiểm tra frame trống, reset nguồn và quỹ đạo đã qua. Test đơn giản với GMC tắt giữ được ID qua một frame trống; clip có frame đen đột ngột với cấu hình GMC hiện tại vẫn đổi ID. Độ chính xác và case thất bại này sẽ được phân tích ở bước 4–5.

### Bước 4 — Đánh giá chất lượng · Tuần 4

1. Xuất kết quả tracking theo định dạng MOTChallenge và chấm bằng TrackEval.
2. Đo **HOTA** để xem chất lượng tổng thể, **IDF1** để xem khả năng giữ ID, **ID switches** để đếm lần đổi ID và **FPS** để đo tốc độ.
3. Ghi rõ phần cứng, độ phân giải và các bước được tính vào thời gian xử lý.
4. Lưu ít nhất 5 đoạn video lỗi để phân biệt bỏ sót người với ghép nhầm ID.

**Hoàn thành khi:** có kết quả thực đo và giải thích được các lỗi chính.

**Trạng thái 18/11/2026:** hoàn thành. Đã xuất MOT TXT, chấm TrackEval và tạo 5 clip lỗi trên 525 frame MOT17-09 + 200 frame đầu MOT17-04 (tập phát triển). Kết quả gộp: HOTA **42,54**, IDF1 **46,13**, **33** lần đổi ID, **12,43 FPS** trên CPU i5-14400. Bỏ sót **8.209** lượt người là vấn đề nổi bật; chưa kết luận riêng detector hay tracker gây ra nếu chưa có thử nghiệm tách yếu tố. Có 29 test đạt. Xem [báo cáo baseline](docs/REPORT.md); MOT17-11/13 vẫn được giữ lại cho đánh giá cuối.

### Bước 5 — Cải thiện tracking · Tuần 5

1. Chỉnh từng nhóm tham số: ngưỡng phát hiện, ngưỡng ghép đối tượng và thời gian giữ track khi mất quan sát.
2. Thử bật ReID trong TrackTrack để dùng thêm đặc trưng ngoại hình; ghi rõ mô hình tạo đặc trưng được dùng.
3. Nếu camera chuyển động, thử bật bù chuyển động camera trong TrackTrack.
4. So sánh trước/sau trên cùng dữ liệu; mỗi lần chỉ thay một yếu tố.
5. Chốt cấu hình rồi chạy trên các video giữ lại để đánh giá cuối.

**Hoàn thành khi:** chọn được cấu hình có đánh đổi chất lượng/tốc độ phù hợp, có số liệu giải thích lựa chọn.

**Trạng thái 18/11/2026:** hoàn thành. Đã thử 12 cấu hình, chọn bằng HOTA trên 1.625 frame phát triển và đóng băng trước khi đánh giá toàn bộ 1.650 frame MOT17-11/13. Trên tập giữ lại, HOTA **39,32 → 46,94**, IDF1 **43,89 → 55,92**; FPS CPU **12,34 → 6,73**. Mặc định dùng ảnh 960, ngưỡng tracker 0.3/0.1/0.3, match 0.8, buffer 60; ReID tắt vì thử nghiệm làm điểm giảm, GMC giữ `sparseOptFlow`. Bỏ sót giảm nhưng FP và đổi ID tăng; case frame đen vẫn mất ID. Có 35 test đạt. Xem [báo cáo và giới hạn](docs/REPORT.md), [quyết định chọn cấu hình](configs/experiments/mot17_selected.json).

### Bước 6 — Hoàn thiện demo và báo cáo · Tuần 6

1. Tạo lệnh chạy với đường dẫn video, cấu hình và thư mục đầu ra.
2. Chuẩn bị demo cảnh thông thường và cảnh đông người.
3. Viết README hướng dẫn cài đặt, chạy và đánh giá.
4. Báo cáo kết quả, ví dụ mất ID và những tình huống hệ thống còn xử lý kém.

**Hoàn thành khi:** người khác có thể làm theo README để chạy lại demo.

**Trạng thái 18/11/2026:** hoàn thành. Hai clip 150 frame từ MOT17-09/04 đã được dùng để kiểm chứng video H.264 trên Chrome, tải MOT/JSONL và snapshot cấu hình. Đã kiểm tra đủ frame, 17 liên kết tài nguyên, bố cục desktop/mobile và 39 test đạt. Trang tĩnh đã được gỡ. README hướng dẫn chạy video bằng CLI. Xem [báo cáo cuối](REPORT.md).

## 4. Tiêu chí hoàn thành dự án

- [x] Chạy được toàn bộ video với YOLO26 + TrackTrack. Đã có toàn bộ 14 video MOT17 train/test, 11.235 frame không lặp ba hậu tố; bộ đếm tracker và video khớp số frame nguồn. Bản H.264 cũng được đọc lại đủ frame; xem [kiểm chứng toàn bộ dữ liệu](outputs/checks/mot17_full_validation.json).
- [x] Xuất được video có khung bao, ID, quỹ đạo và tệp kết quả tracking. Đã kiểm tra video, preview, MOT TXT/JSONL từng frame, ID/tọa độ và các test hiển thị/quỹ đạo.
- [x] Có đánh giá định lượng trên dữ liệu có nhãn, ghi rõ giới hạn nếu mô hình từng học dữ liệu đó. Chấm lại bằng TrackEval trên 1.650 frame giữ lại cho kết quả khớp: HOTA 46,936, IDF1 55,919; GT gốc và prediction khớp checksum. Checkpoint tham chiếu COCO; chưa xác minh độc lập trùng ảnh với MOT17.
- [x] Có cấu hình, phiên bản thư viện và hướng dẫn chạy lại. Cấu hình/trọng số khớp checksum; 43 phiên bản khớp `requirements.txt`, `pip check` đạt.

**Xác minh 18/11/2026:** đạt **4/4 tiêu chí bàn giao trong phạm vi MOT17**, có **39 test đạt**. Bằng chứng tại [project_verification.json](outputs/checks/project_verification.json), [kết quả chấm lại](outputs/evaluations/completion_verification/metrics.json) và [log test](outputs/checks/completion_tests.log).

**Giới hạn nghiệm thu:** mục tiêu giữ ID khi đông người/che khuất mới đạt một phần: còn 95 lần đổi ID, 8.801 lượt bỏ sót và 2.676 dự đoán thừa trên tập giữ lại; case frame đen vẫn mất ID. Tốc độ CPU 6,726 FPS; kế hoạch chưa đặt ngưỡng HOTA/IDF1/FPS tối thiểu để nghiệm thu chất lượng. Ở thời điểm nghiệm thu, MOT20 chưa được bổ sung và tập giữ lại thuộc `MOT17/train`. Đã kiểm tra môi trường hiện có, chưa cài mới toàn bộ trên máy khác.

**Đầu ra toàn bộ MOT17 bổ sung 18/11/2026:** đã xuất 7 video train/5.316 frame và 7 video test/5.919 frame, đủ 42 tên MOT TXT theo DPM/FRCNN/SDP; mọi ảnh trùng đã đối chiếu SHA-256. ZIP train/test và metric toàn bộ train nằm tại `outputs/mot17_full/`. Công cụ batch có 43 test đạt. Xem chi tiết trong [báo cáo](REPORT.md).

## 5. Tài liệu sử dụng

- [YOLO26](https://docs.ultralytics.com/models/yolo26/).
- [TrackTrack và cấu hình tracking trong Ultralytics](https://docs.ultralytics.com/modes/track/).
- [TrackEval](https://github.com/JonathonLuiten/TrackEval).
- [MOT17](https://motchallenge.net/data/MOT17/) và [MOT20](https://motchallenge.net/data/MOT20/).
