"""Encode H.264 previews for browsers."""

import subprocess


def browser_video(source, destination):
    """Keep the original tracking export and make an H.264 preview for browsers."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                             "-i", str(source), "-an", "-c:v", "libx264", "-threads", "2",
                             "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not create browser video: {result.stderr.strip()}")
