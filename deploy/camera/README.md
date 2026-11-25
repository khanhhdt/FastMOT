# Triển khai camera IP nội bộ

Camera dùng YOLO26s OpenVINO FP32 ở `imgsz=1280` và TrackTrack với YOLO26n-ReID. Mục tiêu là 4 FPS tracking và 30 FPS hiển thị box bằng optical flow.

Để chạy trực tiếp hoặc mô phỏng, chuẩn bị [môi trường Python cho CPU](../../README.md#tạo-lại-môi-trường-cpu) bằng `requirements.txt` ở thư mục gốc.

## 1. Chạy với camera IP

Chạy các lệnh dưới đây từ thư mục gốc dự án.

1. Lấy URL RTSP của camera và dùng tài khoản chỉ có quyền xem luồng. Ví dụ URL có dạng `rtsp://viewer:PASSWORD@192.168.1.50:554/stream`.

2. Tạo tệp cấu hình và giới hạn quyền đọc, sửa cho chủ sở hữu tệp.

   ```bash
   mkdir -p .runtime/camera
   touch .runtime/camera/rtsp-url
   chmod 600 .runtime/camera/rtsp-url
   nano .runtime/camera/rtsp-url
   ```

   Dán URL RTSP vào một dòng duy nhất. Nhấn `Ctrl+O`, `Enter` để lưu rồi `Ctrl+X` để thoát.

3. Khởi động trang camera.

   ```bash
   python scripts/web_tools/serve_camera.py \
     --rtsp-url-file .runtime/camera/rtsp-url
   ```

4. Mở [localhost:8003](http://127.0.0.1:8003) để xem ảnh, box, ID và các chỉ số FPS.

## 2. Chạy mô phỏng

Cần FFmpeg, MediaMTX và clip mẫu. Lệnh dưới phát clip qua RTSP trên localhost, thử ngắt và nối lại luồng, rồi chạy tiếp trong 3 phút.

```bash

python scripts/web_tools/verify_camera.py \
  --mediamtx .runtime/camera-test/mediamtx \
  --fixture deploy/video/fixture.mp4 \
  --seconds 180 --output outputs/camera_sim
```

Mở [localhost:8003](http://127.0.0.1:8003) khi terminal báo `Viewer is live`.

## 3. Chuẩn bị triển khai

Chạy trên máy chủ Linux có systemd, quyền `sudo`, Python 3.11 có `venv` và [Caddy chạy bằng systemd](https://caddyserver.com/docs/running#linux-service).

### 3.1. Chép bản phát hành vào `/opt`

Chọn phiên bản chưa tồn tại, ví dụ `v1`.

```bash
MOT_RELEASE=/opt/mot/releases/v1
sudo install -d -m 0755 /opt/mot/releases
sudo mkdir -m 0755 "$MOT_RELEASE" &&
  sudo install -d -m 0755 "$MOT_RELEASE/deploy" "$MOT_RELEASE/weights" &&
  sudo cp -R src scripts configs requirements.txt "$MOT_RELEASE/" &&
  sudo cp -R deploy/camera "$MOT_RELEASE/deploy/" &&
  sudo cp -R weights "$MOT_RELEASE/weights/" &&
  sudo /usr/bin/python3.11 -m venv "$MOT_RELEASE/.venv" &&
  sudo "$MOT_RELEASE/.venv/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu &&
  sudo "$MOT_RELEASE/.venv/bin/python" -m pip install -r "$MOT_RELEASE/requirements.txt" &&
  sudo ln -sfnT "$MOT_RELEASE" /opt/mot/current
```

### 3.2. Cài dịch vụ camera

```bash
getent group mot >/dev/null || sudo groupadd --system mot
id -u mot >/dev/null 2>&1 || sudo useradd --system --gid mot --no-create-home --shell /usr/sbin/nologin mot
sudo install -d -m 0755 /etc/mot
sudo touch /etc/mot/rtsp-url
sudo chown root:root /etc/mot/rtsp-url
sudo chmod 600 /etc/mot/rtsp-url
sudoedit /etc/mot/rtsp-url
```

Dán URL RTSP của tài khoản chỉ đọc vào một dòng, lưu tệp rồi chạy:

```bash
sudo install -m 0644 deploy/camera/mot-camera.service /etc/systemd/system/mot-camera.service
sudo systemctl daemon-reload
sudo systemctl restart mot-camera
sudo journalctl -u mot-camera -n 30 --no-pager
```

### 3.3. Mở trang camera qua Caddy

```bash
caddy hash-password
sudo touch /etc/caddy/mot-camera-users.caddy
sudo chown root:caddy /etc/caddy/mot-camera-users.caddy
sudo chmod 640 /etc/caddy/mot-camera-users.caddy
sudoedit /etc/caddy/mot-camera-users.caddy
```

Nhập mật khẩu xem trang khi Caddy hỏi. Trong tệp vừa mở, lưu một dòng `viewer <hash>`, thay `<hash>` bằng giá trị lệnh in ra.

```bash
sudo install -m 0644 deploy/camera/Caddyfile /etc/caddy/mot-camera.caddy
sudoedit /etc/caddy/Caddyfile
```

Thêm dòng `import /etc/caddy/mot-camera.caddy` ở ngoài các khối cấu hình, lưu tệp rồi chạy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile &&
sudo systemctl reload caddy
```

Trỏ `mot.home.arpa` về IP máy chủ trong DNS nội bộ. Nếu dùng máy khách Linux và chưa có DNS nội bộ, chạy `sudoedit /etc/hosts` trên máy khách rồi thêm dòng `192.168.1.10 mot.home.arpa` với IP thật của máy chủ. Cài chứng chỉ CA `/var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt` lấy từ máy chủ vào kho tin cậy của máy khách. Giới hạn cổng HTTPS cho LAN. [Hướng dẫn CA nội bộ của Caddy](https://caddyserver.com/docs/running#local-https-with-systemd)

## 4. Nghiệm thu

- Mở [mot.home.arpa](https://mot.home.arpa). Khi chưa đăng nhập, trang và API phải trả `401`. Sau khi đăng nhập, kiểm tra luồng ảnh, box, ID và trạng thái camera.
- Chạy camera thật ít nhất 30 phút, gồm lúc đông người và một lần mất mạng. Tracking cần đạt ít nhất **3 FPS trong 95% cửa sổ một phút có tín hiệu**. Thời gian nối lại tối đa **15 giây** sau khi camera phát lại.
- Đo độ trễ từ cảnh thật tới trình duyệt bằng đồng hồ trong khung hình. Mục tiêu **p95 không quá 1,2 giây**. Sau 24 giờ chạy ổn định, bật tự khởi động bằng `systemctl enable mot-camera`.
- Khi thay model hoặc tracker, chấm lại trên `deploy/video/fixture.mp4`. Mức giảm tối đa so với bản tham chiếu là **2 điểm HOTA** và **3 điểm IDF1**.
