# Báo cáo cải thiện hệ thống theo dõi người

Hệ thống dùng YOLO26s để phát hiện người và TrackTrack để gán ID qua các frame. Các bước dưới đây mô tả quá trình cải thiện từ cấu hình ban đầu đến profile `reid` đã dùng để xuất MOT17 test.

## 1. Đo cấu hình ban đầu

Chạy YOLO26s ở kích thước đầu vào 640 và chấm TrackTrack với ground truth (nhãn khung bao và ID chuẩn). Trên 725 frame của MOT17-04 và 09, baseline đạt **HOTA 42,54%** và **IDF1 46,13%**. Kết quả cho thấy hệ thống bỏ sót nhiều người và có các ca đổi ID khi người đi cắt nhau. HOTA phản ánh cả chất lượng phát hiện lẫn ghép track. IDF1 tập trung vào việc gán đúng ID. [Điểm baseline và clip lỗi](../outputs/evaluations/baseline_mot17/metrics.json)

## 2. Cải thiện phát hiện người và xử lý chuyển cảnh

Thử lần lượt các ngưỡng phát hiện, tạo track và ghép track. Tăng kích thước đầu vào từ 640 lên 960 ở V1, rồi chọn 1280 cho profile `quality` ở V2 để phát hiện tốt hơn trong cảnh đông. V2 cũng bảo vệ bước bù chuyển động camera (GMC) khi gặp frame đen và đặt lại tracker khi phát hiện chuyển cảnh rõ rệt.

Trên cùng **5.316 frame MOT17 train**, HOTA tăng từ **45,31% ở V1 lên 47,03% ở V2**. Số lượt bỏ sót giảm từ **51.401 xuống 48.285**. V2 có thêm profile `fast` dành cho tốc độ và giao diện tải video lên để theo dõi tiến độ, xem và tải kết quả. [Điểm V1](../outputs/mot17_full/evaluation_train/metrics.json) · [Điểm V2](../outputs/mot_full_v2/mot17/evaluation_train/metrics.json) · [Kiểm chứng V2](../outputs/improvements/verification.json)

## 3. Thêm ReID để ghép lại người đã mất dấu

Giữ track mất dấu trong 60 frame và dùng mô hình ReID riêng để so ngoại hình khi người xuất hiện lại. Trên toàn bộ MOT17 train, cấu hình này đạt **HOTA 48,23%**, **IDF1 58,73%** và **378 lần đổi ID**. Đây là mốc để đo tác động của các bước nối quỹ đạo sau đó. [Cấu hình TrackTrack + ReID](../configs/trackers/tracktrack_quality_reid60.yaml) · [Benchmark ReID](../outputs/mot17_train_full_benchmark/benchmark.json)

## 4. Nối quỹ đạo ở camera cố định

So ngoại hình, vị trí và hướng đi ở hai đầu khoảng mất dấu để nối track mới về ID cũ. Với người đứng hoặc ngồi lâu ở một vị trí, dùng thêm vùng neo đã kiểm tra thủ công trên MOT17-02 và 04. Nối theo tọa độ ảnh cũng được áp dụng trên MOT17-09. Ba video camera cố định tăng IDF1 từ **59,05% lên 59,56%** và giảm số lần đổi ID từ **178 xuống 173**. [Kết quả nhóm camera cố định](../outputs/mot17_train_benchmark/benchmark.json)

## 5. Kiểm tra box điểm thấp và các ngưỡng của tracker

Lưu box YOLO trước khi qua TrackTrack trên MOT17-02 và 10. Trong **105 đầu khoảng che khuất thiếu track khớp với ground truth**, có **66 đầu đã có box đủ điều kiện để đem ra ghép**. Kết quả này cho thấy cần kiểm tra cách tracker ghép và xác nhận track bên cạnh khả năng phát hiện của YOLO. [Đối chiếu detection và tracking](../outputs/mot17_detection_audit/audit.html)

Đã thử nới ngưỡng ghép lại track mất dấu, tăng bộ nhớ, hạ ngưỡng box và xác nhận track sớm hơn. Một số thay đổi làm tăng ghép nhầm hoặc báo nhầm. Lợi ích cũng khác nhau giữa các video. Kết quả kiểm chứng cho thấy nên giữ ngưỡng tạo track ở **0,45**, bộ nhớ **60 frame** và xác nhận track sau **5 frame**. [Thử khôi phục track mất dấu](../outputs/mot17_tracktrack_rebind/benchmark.html) · [Thử box điểm thấp và xác nhận track](../outputs/mot17_tracktrack_low_and_new/benchmark.html) · [Kiểm chứng ngưỡng 0,35](../outputs/mot17_spawn_035_validation/benchmark.html)

## 6. Bù chuyển động camera trước khi nối quỹ đạo

Với camera di chuyển, dùng GMC để đưa vị trí chân và vận tốc ở hai đầu quỹ đạo về cùng hệ tọa độ, rồi mới so chuyển động và ngoại hình. Trên MOT17-05, 10, 11 và 13, IDF1 gộp tăng từ **58,20%** với ReID gốc lên **58,76%** khi nối theo tọa độ ảnh và **59,04%** khi nối sau bù camera. Số lần đổi ID giảm từ **200 xuống 196**. Phần cải thiện thêm so với nối theo tọa độ ảnh đến từ MOT17-10. [So sánh ba cấu hình](../outputs/mot17_train_reid_batch/comparison.html)

## 7. Đưa cấu hình đã chọn vào batch và chấm lại MOT17 train

Profile `reid` chạy YOLO26s 1280, TrackTrack + ReID và chọn nhánh nối theo loại camera. MOT17-02, 04 và 09 dùng nối theo tọa độ ảnh. MOT17-05, 10, 11 và 13 dùng nối sau bù camera.

Trên toàn bộ **7 video và 5.316 frame MOT17 train**, cấu hình cuối đạt **HOTA 48,58%**, **IDF1 59,36%** và **369 lần đổi ID**. So với hệ thống ngay trước khi bổ sung nối cho camera di chuyển, số ca giữ đúng ID qua các khoảng che khuất tăng từ **121 lên 125 trên 338 ca**. [Điểm TrackEval](../outputs/mot17_train_reid_batch/evaluation_train/metrics.json) · [Video và tệp MOT của tập train](../outputs/mot17_train_reid_batch/index.html)

## 8. Kiểm tra trên MOT20 train và xuất MOT17 test

Chạy cấu hình đã chốt trên **4 video và 8.931 frame MOT20 train**. So với V2, HOTA tăng từ **24,78% lên 25,56%**, IDF1 tăng từ **29,98% lên 31,11%** và số lần đổi ID giảm từ **609 xuống 575**. Bước nối ghi nhận **9 mối đúng và 1 mối ghép nhầm** qua đối chiếu ground truth. MOT20-05 vẫn là cảnh khó nhất, với khoảng **5,6% lượt người trong ground truth** khớp với đầu ra tracking. [Báo cáo MOT20 train](../outputs/mot20_train_reid_benchmark/benchmark.html)

Đã xuất đủ **7 video, 5.919 frame và 21 tệp MOT TXT** cho MOT17 test. MOT17-01, 03 và 08 dùng nối theo tọa độ ảnh. MOT17-06, 07, 12 và 14 dùng nối sau bù camera. Bộ test được dùng để xem kết quả dự đoán trên các cảnh mới. Các điểm chất lượng trong báo cáo đến từ những tập có ground truth. [Trang xem MOT17 test](../outputs/mot17_test_reid_batch/index.html) · [ZIP tệp MOT của tập test](../outputs/mot17_test_reid_batch/motchallenge/test.zip)
