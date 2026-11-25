const camera = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const connection = document.querySelector("#connection");
const placeholder = document.querySelector("#placeholder");
const fps = document.querySelector("#fps");
const displayFps = document.querySelector("#display-fps");
const age = document.querySelector("#age");
const labels = {
  connecting: "Đang kết nối",
  live: "Đang phát",
  stale: "Mất tín hiệu",
  reconnecting: "Đang kết nối lại",
  failed: "Dịch vụ gặp lỗi",
};
let frameUrl;
let etag;
let displaySession;
let overlay;
let overlaySeenAt;
let lastStatusAt = -Infinity;
let targetDisplayFps = 15;
const displayedAt = [];

function drawOverlay() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const ctx = canvas.getContext("2d");
  if (!overlay || overlay.session !== displaySession ||
      overlay.age_ms + performance.now() - overlaySeenAt > 800 ||
      !camera.naturalWidth || !camera.naturalHeight) return;
  const scale = Math.min(canvas.width / overlay.width, canvas.height / overlay.height);
  const left = (canvas.width - overlay.width * scale) / 2;
  const top = (canvas.height - overlay.height * scale) / 2;
  for (const track of overlay.tracks) {
    const [x1, y1, x2, y2] = track.xyxy;
    const x = left + x1 * scale;
    const y = top + y1 * scale;
    const w = (x2 - x1) * scale;
    const h = (y2 - y1) * scale;
    const color = `hsl(${track.id * 47 % 360} 85% 58%)`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * ratio;
    ctx.strokeRect(x, y, w, h);
    const label = `ID ${track.id}`;
    ctx.font = `${14 * ratio}px sans-serif`;
    const labelHeight = 20 * ratio;
    const labelWidth = ctx.measureText(label).width + 8 * ratio;
    const labelTop = Math.max(0, y - labelHeight);
    ctx.fillStyle = color;
    ctx.fillRect(x, labelTop, labelWidth, labelHeight);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 4 * ratio, labelTop + 15 * ratio);
  }
}

camera.onload = () => {
  const now = performance.now();
  displayedAt.push(now);
  while (displayedAt.length && now - displayedAt[0] > 5000) displayedAt.shift();
  if (displayedAt.length > 1) {
    const rate = 1000 * (displayedAt.length - 1) / (now - displayedAt[0]);
    displayFps.textContent = `${rate.toFixed(1)} / ${targetDisplayFps} FPS`;
  }
  drawOverlay();
};
window.addEventListener("resize", drawOverlay);

function setState(state) {
  connection.textContent = labels[state];
  connection.dataset.live = String(state === "live");
  if (state !== "live") {
    camera.hidden = canvas.hidden = true;
    camera.removeAttribute("src");
    if (frameUrl) URL.revokeObjectURL(frameUrl);
    frameUrl = etag = displaySession = overlay = undefined;
    displayedAt.length = 0;
    placeholder.hidden = false;
    placeholder.textContent = state === "connecting" ? "Đang chờ hình ảnh từ camera" : labels[state];
    age.textContent = displayFps.textContent = "—";
    drawOverlay();
  }
}

async function get(path, options) {
  return fetch(path, { cache: "no-store", signal: AbortSignal.timeout(2000), ...options });
}

async function update() {
  const start = performance.now();
  try {
    if (start - lastStatusAt >= 250) {
      const response = await get("/api/camera/status");
      if (!response.ok) throw new Error("status unavailable");
      const status = await response.json();
      lastStatusAt = start;
      setState(status.state);
      fps.textContent = `${status.effective_fps.toFixed(1)} / ${status.target_fps} FPS`;
      targetDisplayFps = status.display_target_fps;
      age.textContent = status.state === "live" ? `${status.frame_age_ms} ms` : "—";
      overlay = status.overlay;
      overlaySeenAt = performance.now();
      drawOverlay();
    }
    const response = await get("/api/camera/live.jpg", { headers: etag ? { "If-None-Match": etag } : {} });
    if (response.status === 503) {
      if (!camera.hidden) setState("stale");
    } else if (response.status !== 304) {
      if (!response.ok) throw new Error("frame unavailable");
      const nextUrl = URL.createObjectURL(await response.blob());
      camera.src = nextUrl;
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = nextUrl;
      etag = response.headers.get("ETag");
      displaySession = etag?.slice(1, -1).replace(/-\d+$/, "");
      camera.hidden = canvas.hidden = false;
      placeholder.hidden = true;
      setState("live");
    }
  } catch {
    setState("reconnecting");
    fps.textContent = "—";
  } finally {
    setTimeout(update, Math.max(0, 1000 / targetDisplayFps - (performance.now() - start)));
  }
}

update();
