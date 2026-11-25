const flowVideo = document.querySelector("#flow-video");
const connection = document.querySelector("#connection");
const placeholder = document.querySelector("#placeholder");
const fps = document.querySelector("#fps");
const displayFps = document.querySelector("#display-fps");
const flowFps = document.querySelector("#flow-fps");
const age = document.querySelector("#age");
const labels = {
  connecting: "Đang kết nối",
  live: "Đang phát",
  stale: "Mất tín hiệu",
  reconnecting: "Đang kết nối lại",
  failed: "Dịch vụ gặp lỗi",
};
let targetDisplayFps = 30;
let stream;
const displayedAt = [];

function frameShown(now) {
  displayedAt.push(now);
  while (displayedAt.length && now - displayedAt[0] > 5000) displayedAt.shift();
  if (displayedAt.length > 1) {
    const rate = 1000 * (displayedAt.length - 1) / (now - displayedAt[0]);
    displayFps.textContent = `${rate.toFixed(1)} / ${targetDisplayFps} FPS`;
  }
}

function setState(state) {
  connection.textContent = labels[state];
  connection.dataset.live = String(state === "live");
  if (state !== "live") {
    stream?.controller.abort();
    stream = undefined;
    flowVideo.hidden = true;
    displayedAt.length = 0;
    placeholder.hidden = false;
    placeholder.textContent = state === "connecting" ? "Đang chờ hình ảnh từ camera" : labels[state];
    age.textContent = displayFps.textContent = flowFps.textContent = "—";
  }
}

async function get(path, options) {
  return fetch(path, { cache: "no-store", signal: AbortSignal.timeout(2000), ...options });
}

async function playStream(session) {
  const controller = new AbortController();
  const { signal } = controller;
  const active = { controller, session };
  stream = active;
  let pending;
  let decoding = false;
  const frames = [];
  const interval = 1000 / targetDisplayFps;
  let nextPaint;
  let animation;

  function paint(now) {
    animation = requestAnimationFrame(paint);
    if (now + 2 < nextPaint) return;
    nextPaint += interval;
    if (nextPaint < now) nextPaint = now + interval;
    const bitmap = frames.shift();
    if (!bitmap) return;
    try {
      if (flowVideo.width !== bitmap.width || flowVideo.height !== bitmap.height) {
        flowVideo.width = bitmap.width;
        flowVideo.height = bitmap.height;
      }
      flowVideo.getContext("2d").drawImage(bitmap, 0, 0);
      flowVideo.hidden = false;
      placeholder.hidden = true;
      frameShown(now);
    } finally {
      bitmap.close();
    }
  }

  signal.addEventListener("abort", () => {
    cancelAnimationFrame(animation);
    for (const bitmap of frames) bitmap.close();
    frames.length = 0;
  }, { once: true });

  async function decodeLatest() {
    decoding = true;
    try {
      while (pending && !signal.aborted) {
        const jpeg = pending;
        pending = undefined;
        const bitmap = await createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }));
        if (signal.aborted) {
          bitmap.close();
          return;
        }
        frames.push(bitmap);
        if (frames.length > 4) frames.shift().close();
        if (animation === undefined) {
          // Give three frames time to arrive before starting the 30 FPS playout clock.
          nextPaint = performance.now() + 3 * interval;
          animation = requestAnimationFrame(paint);
        }
      }
    } catch {
      controller.abort();
    } finally {
      decoding = false;
    }
  }

  try {
    const response = await get("/api/camera/stream", { signal });
    if (!response.ok) return;
    const reader = response.body.getReader();
    let chunk = new Uint8Array();
    let offset = 0;
    async function readBytes(length) {
      const result = new Uint8Array(length);
      for (let read = 0; read < length;) {
        if (offset === chunk.length) {
          const { done, value } = await reader.read();
          if (done) return null;
          chunk = value;
          offset = 0;
        }
        const count = Math.min(chunk.length - offset, length - read);
        result.set(chunk.subarray(offset, offset + count), read);
        offset += count;
        read += count;
      }
      return result;
    }
    while (!signal.aborted) {
      const header = await readBytes(4);
      if (!header) break;
      const length = new DataView(header.buffer).getUint32(0);
      if (length === 0 || length > 4 * 1024 * 1024) break;
      const jpeg = await readBytes(length);
      if (!jpeg) break;
      pending = jpeg;
      if (!decoding) void decodeLatest();
    }
  } catch {
    // Status polling will restart the stream on disconnect.
  } finally {
    controller.abort();
    if (stream === active) setState("reconnecting");
  }
}

async function update() {
  const start = performance.now();
  try {
    const response = await get("/api/camera/status");
    if (!response.ok) throw new Error("status unavailable");
    const status = await response.json();
    targetDisplayFps = status.display_target_fps;
    setState(status.state);
    if (status.state === "live" && stream?.session !== status.session) {
      if (stream) setState("connecting");
      void playStream(status.session);
    }
    fps.textContent = `${status.effective_fps.toFixed(1)} / ${status.target_fps} FPS`;
    flowFps.textContent = status.state === "live" ? `${status.flow_fps.toFixed(1)} FPS` : "—";
    age.textContent = status.state === "live" ? `${status.frame_age_ms} ms` : "—";
  } catch {
    setState("reconnecting");
    fps.textContent = "—";
  } finally {
    setTimeout(update, Math.max(0, 250 - (performance.now() - start)));
  }
}

update();
