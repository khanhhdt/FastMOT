const $ = id => document.getElementById(id);
let file = null, upload = null, activeJob = null, polling = null, maxBytes = 1024 ** 3;
const states = {uploading:'Đang tải lên',queued:'Đang chờ',running:'Đang xử lý',done:'Hoàn tất',failed:'Có lỗi',cancelled:'Đã hủy'};
const terminal = new Set(['done','failed','cancelled']);

function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Yêu cầu không thành công');
  return data;
}
function choose(selected) {
  file = selected || null;
  error('');
  if (file && (!file.size || file.size > maxBytes)) {
    error('Chọn video lớn hơn 0 byte và không quá 1 GiB.'); file = null;
  }
  $('file-label').textContent = file ? `${file.name} · ${(file.size/1024/1024).toFixed(1)} MiB` : 'MP4, MOV, MKV, AVI, WebM · Tối đa 1 GiB';
  $('start').disabled = !file || !!upload;
}
$('choose-file').onclick = () => $('video-file').click();
$('video-file').onchange = () => choose($('video-file').files[0]);
for (const event of ['dragenter','dragover']) $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const event of ['dragleave','drop']) $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', e => choose(e.dataTransfer.files[0]));

function render(job) {
  activeJob = job.id;
  $('state').textContent = states[job.state] || job.state;
  $('job-name').textContent = job.filename;
  $('status-detail').textContent = job.message;
  $('frame-detail').textContent = job.total ? `${job.frames.toLocaleString()} / ${job.total.toLocaleString()} frame` : '';
  if (job.total && job.frames) $('progress').value = Math.min(1, job.frames/job.total);
  else if (!terminal.has(job.state)) $('progress').removeAttribute('value');
  else $('progress').value = job.state === 'done' ? 1 : 0;
  $('cancel').hidden = terminal.has(job.state);
  $('cancel').disabled = false;
  if (job.state === 'failed') error(job.message);
  if (job.state === 'done') {
    $('result').hidden = false;
    if ($('result-video').dataset.job !== job.id) {
      $('result-video').dataset.job = job.id;
      $('result-video').src = `/api/jobs/${job.id}/files/preview.mp4`;
      $('downloads').replaceChildren();
      for (const [filename,label] of [['preview.mp4','Tải video MP4'],['tracks.txt','MOT TXT'],['tracks.jsonl','Tracking JSONL'],['summary.json','Thống kê chi tiết']]) {
        const link = document.createElement('a');
        link.href = `/api/jobs/${job.id}/files/${filename}?download=1`;
        link.textContent = label; link.download = filename;
        $('downloads').append(link);
      }
    }
    $('result-stats').textContent = `${job.frames.toLocaleString()} frame · ${job.fps.toFixed(2)} FPS xử lý · ${job.observations.toLocaleString()} lượt quan sát tracking`;
  } else {
    $('result-video').pause(); $('result').hidden = true;
  }
}
async function watch(id) {
  clearTimeout(polling);
  if (activeJob !== id) return;
  try {
    const job = await api(`/api/jobs/${id}`);
    if (activeJob !== id) return;
    render(job);
    if (!terminal.has(job.state)) polling = setTimeout(() => watch(id), 700);
    else await loadHistory();
  } catch (e) { error(e.message); }
}
function openJob(id) { if (upload) return; error(''); activeJob = id; location.hash = `job=${id}`; watch(id); }
async function loadHistory() {
  const data = await api('/api/jobs');
  $('history-body').replaceChildren();
  $('history-empty').hidden = data.jobs.length > 0;
  $('history-table').hidden = !data.jobs.length;
  for (const job of data.jobs) {
    const row = document.createElement('tr');
    for (const value of [job.filename,job.profile === 'fast' ? 'Nhanh' : 'Chất lượng',states[job.state] || job.state]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    const cell = document.createElement('td'), button = document.createElement('button');
    button.type = 'button'; button.className = 'text-button'; button.textContent = 'Mở';
    button.onclick = () => openJob(job.id); cell.append(button); row.append(cell);
    $('history-body').append(row);
  }
}
$('refresh').onclick = () => loadHistory().catch(e => error(e.message));
$('upload-form').onsubmit = e => {
  e.preventDefault();
  if (!file || upload) return;
  error(''); clearTimeout(polling); $('start').disabled = true;
  $('result-video').pause(); $('result').hidden = true;
  $('state').textContent = 'Đang tải lên'; $('job-name').textContent = file.name;
  $('status-detail').textContent = 'Đang gửi video tới máy chủ cục bộ';
  $('frame-detail').textContent = ''; $('progress').value = 0; $('cancel').hidden = false;
  const query = new URLSearchParams({filename:file.name,profile:$('profile').value});
  if ($('frame-limit').value) query.set('frames',$('frame-limit').value);
  const xhr = upload = new XMLHttpRequest();
  xhr.open('POST', `/api/jobs?${query}`); xhr.setRequestHeader('Content-Type','application/octet-stream');
  xhr.upload.onprogress = e => { if (e.lengthComputable) { $('progress').value = e.loaded/e.total; $('frame-detail').textContent = `${Math.round(e.loaded/e.total*100)}% tải lên`; } };
  xhr.onload = () => {
    upload = null; $('start').disabled = !file;
    try { const data = JSON.parse(xhr.responseText); if (xhr.status !== 202) throw new Error(data.error || 'Không thể tạo job'); openJob(data.id); loadHistory().catch(e => error(e.message)); }
    catch (e) { error(e.message); $('state').textContent = 'Có lỗi'; $('cancel').hidden = true; }
  };
  xhr.onerror = () => { upload = null; $('start').disabled = !file; error('Không kết nối được máy chủ. Kiểm tra server rồi thử lại.'); $('cancel').hidden = true; };
  xhr.onabort = () => { upload = null; $('start').disabled = !file; $('state').textContent = 'Đã hủy tải lên'; $('cancel').hidden = true; };
  xhr.send(file);
};
$('cancel').onclick = async () => {
  if (upload) { upload.abort(); return; }
  if (!activeJob) return;
  const id = activeJob;
  $('cancel').disabled = true;
  try { const job = await api(`/api/jobs/${id}/cancel`, {method:'POST'}); if (activeJob === id) { render(job); watch(id); } }
  catch (e) { error(e.message); $('cancel').disabled = false; }
};
(async () => {
  const data = await api('/api/profiles'); maxBytes = data.max_upload_bytes;
  for (const p of data.profiles) {
    const option = document.createElement('option'); option.value = p.name;
    option.textContent = `${p.label} · ${p.model.replace('.pt','')} · ${p.imgsz}px`;
    $('profile').append(option);
  }
  await loadHistory();
  const match = location.hash.match(/^#job=([0-9a-f]{32})$/); if (match) openJob(match[1]);
})().catch(e => error(e.message));
