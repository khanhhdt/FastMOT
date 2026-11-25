// Measure frames actually presented by Chrome's WebRTC video element.
const [port, seconds] = process.argv.slice(2).map(Number);
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = targets.find((target) => target.type === "page" && target.url.includes("/camera"));
if (!page) throw new Error("WebRTC page is not open in Chrome");
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});
const expression = `new Promise((resolve, reject) => {
  const deadline = performance.now() + 30000;
  function ready() {
    const video = document.querySelector('video');
    if (!video || video.readyState < 3 || !video.requestVideoFrameCallback) {
      if (performance.now() > deadline) return reject(new Error('WebRTC video did not start'));
      return setTimeout(ready, 200);
    }
    const before = video.getVideoPlaybackQuality();
    const started = performance.now();
    let callbacks = 0;
    let first, last;
    let previousCallback;
    const gaps = [];
    function next(_, metadata) {
      const now = performance.now();
      if (previousCallback !== undefined) gaps.push(now - previousCallback);
      previousCallback = now;
      callbacks++;
      first ??= metadata.presentedFrames;
      last = metadata.presentedFrames;
      if (performance.now() - started < ${seconds * 1000}) video.requestVideoFrameCallback(next);
    }
    video.requestVideoFrameCallback(next);
    setTimeout(() => {
      const after = video.getVideoPlaybackQuality();
      const elapsed = (performance.now() - started) / 1000;
      gaps.sort((a, b) => a - b);
      resolve({elapsed_seconds: elapsed, callback_frames: callbacks,
        presented_frames: last - first + 1,
        presented_fps: callbacks / elapsed,
        display_gap_p95_ms: gaps[Math.floor(.95 * (gaps.length - 1))],
        display_gap_max_ms: gaps[gaps.length - 1],
        dropped_frames: after.droppedVideoFrames - before.droppedVideoFrames,
        width: video.videoWidth, height: video.videoHeight,
        visibility: document.visibilityState});
    }, ${seconds * 1000});
  }
  ready();
})`;
socket.send(JSON.stringify({ id: 1, method: "Runtime.evaluate", params: {
  expression, awaitPromise: true, returnByValue: true
}}));
const result = await new Promise((resolve, reject) => {
  socket.addEventListener("message", (event) => {
    const reply = JSON.parse(event.data);
    if (reply.id !== 1) return;
    if (reply.error || reply.result?.exceptionDetails) reject(reply.error || reply.result.exceptionDetails);
    else resolve(reply.result.result.value);
  });
  socket.addEventListener("error", reject, { once: true });
});
socket.close();
console.log(JSON.stringify(result));
