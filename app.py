import inspect
import io
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pillow_heif
import rawpy
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps
from skimage.segmentation import find_boundaries, slic
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC/HEIFサポートおよびデコンプレッションボム対策
pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = 100_000_000

st.set_page_config(
    page_title="NDVI解析 & セグメンテーションアプリ",
    page_icon="🍓",
    layout="wide",
)

# 定数設定
DISPLAY_WIDTH = 600
BRUSH_DISPLAY_WIDTH = 460
WORK_MAX_SIDE = 1024
MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_URL_HOSTS = {"raw.githubusercontent.com"}

SAMPLE_SEG_NAME = "IMG_3715.jpg"
SAMPLE_SEG_URL = "https://raw.githubusercontent.com/yasaibusisetuengei/plant-level-analysis-of-strawberry/main/IMG_3715.jpg"
SAMPLE_NDVI_NAME = "IMG_3715_NDVI.jpg"
SAMPLE_NDVI_RAW_URL = "https://raw.githubusercontent.com/yasaibusisetuengei/plant-level-analysis-of-strawberry/main/IMG_3715_NDVI.jpg"
SAMPLE_MASK_NAME = "IMG_3715_mask.png"
SAMPLE_MASK_RAW_URL = "https://raw.githubusercontent.com/yasaibusisetuengei/plant-level-analysis-of-strawberry/main/IMG_3715_mask.png"

def _cache_data(**kwargs):
    if st.runtime.exists():
        return st.cache_data(**kwargs)
    return lambda fn: fn


def _cache_resource(**kwargs):
    if st.runtime.exists():
        return st.cache_resource(**kwargs)
    return lambda fn: fn


COLOR_PRESETS = {
    "イチゴNDVI用": ((139, 173), (70, 255), (50, 255)),
    "緑": ((35, 85), (40, 255), (40, 255)),
    "赤": ((170, 10), (70, 255), (50, 255)),
    "青": ((95, 130), (50, 255), (50, 255)),
    "黄": ((18, 35), (70, 255), (70, 255)),
    "白": ((0, 179), (0, 50), (200, 255)),
    "黒": ((0, 179), (0, 255), (0, 50)),
}

MASK_DISPLAY_COLORS = {
    "水色": (0, 200, 255),
    "赤": (255, 60, 0),
    "緑": (0, 220, 80),
    "黄": (255, 220, 0),
    "紫": (180, 60, 255),
}

# ------------------------------------------------------------
# 色相環カスタムコンポーネント (HTML5 Canvas ドラッグ対応)
# ------------------------------------------------------------
HUE_WHEEL_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: sans-serif;
    background: transparent;
    user-select: none;
  }
  canvas {
    cursor: grab;
    touch-action: none;
  }
  canvas.dragging {
    cursor: grabbing;
  }
  .info {
    margin-top: 4px;
    font-size: 13px;
    color: #555;
  }
</style>
</head>
<body>
  <canvas id="wheel" width="240" height="240"></canvas>
  <div class="info" id="info">H: 35 ～ 85 (0～179)</div>
<script>
  const canvas = document.getElementById("wheel");
  const ctx = canvas.getContext("2d");
  const infoEl = document.getElementById("info");

  const cx = 120, cy = 120;
  const rOuter = 104, rInner = 64;
  let hMin = 35;
  let hMax = 85;
  let lastSyncId = -1;
  let dragging = null;
  let dragStartAngle = 0;
  let startHMin = 35, startHMax = 85;

  function sendMessage(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }

  function hToAngle(h) {
    return (h / 180) * Math.PI * 2 - Math.PI / 2;
  }

  function angleToH(angle) {
    let deg = ((angle + Math.PI / 2) * 180 / Math.PI) % 360;
    if (deg < 0) deg += 360;
    return Math.round(deg / 2) % 180;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let deg = 0; deg < 360; deg++) {
      const a0 = (deg - 0.8) * Math.PI / 180 - Math.PI / 2;
      const a1 = (deg + 0.8) * Math.PI / 180 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, rOuter, a0, a1);
      ctx.arc(cx, cy, rInner, a1, a0, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${deg}, 100%, 50%)`;
      ctx.fill();
    }

    const aMin = hToAngle(hMin);
    const aMax = hToAngle(hMax);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, rInner - 4, aMin, aMax, false);
    ctx.closePath();
    ctx.fillStyle = "rgba(0, 160, 255, 0.22)";
    ctx.fill();

    function drawHandle(h, color) {
      const a = hToAngle(h);
      const x2 = cx + Math.cos(a) * (rOuter + 4);
      const y2 = cy + Math.sin(a) * (rOuter + 4);

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
      const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
      ctx.beginPath();
      ctx.arc(hx, hy, 8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }

    drawHandle(hMin, "#ffffff");
    drawHandle(hMax, "#ffeb3b");

    infoEl.textContent = `H (色相): ${hMin} ～ ${hMax}  (ドラッグで範囲調整)`;
  }

  function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function handleDist(h, px, py) {
    const a = hToAngle(h);
    const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
    const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
    return Math.hypot(px - hx, py - hy);
  }

  canvas.addEventListener("pointerdown", (e) => {
    const { x, y } = getPointerPos(e);
    const dMin = handleDist(hMin, x, y);
    const dMax = handleDist(hMax, x, y);

    if (dMin < 18 && dMin <= dMax) {
      dragging = "min";
    } else if (dMax < 18) {
      dragging = "max";
    } else {
      const dist = Math.hypot(x - cx, y - cy);
      if (dist <= rOuter + 10) {
        dragging = "range";
        dragStartAngle = Math.atan2(y - cy, x - cx);
        startHMin = hMin;
        startHMax = hMax;
      }
    }
    if (dragging) {
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add("dragging");
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const { x, y } = getPointerPos(e);
    const angle = Math.atan2(y - cy, x - cx);

    if (dragging === "min") {
      hMin = angleToH(angle);
    } else if (dragging === "max") {
      hMax = angleToH(angle);
    } else if (dragging === "range") {
      const diffRad = angle - dragStartAngle;
      const diffH = Math.round((diffRad / (Math.PI * 2)) * 180);
      hMin = ((startHMin + diffH) % 180 + 180) % 180;
      hMax = ((startHMax + diffH) % 180 + 180) % 180;
    }
    draw();
  });

  function endDrag() {
    if (!dragging) return;
    dragging = null;
    canvas.classList.remove("dragging");
    sendMessage("streamlit:setComponentValue", {
      value: { h_min: hMin, h_max: hMax },
      dataType: "json",
    });
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "streamlit:render") {
      const args = event.data.args || {};
      if (typeof args.sync_id === "number" && args.sync_id !== lastSyncId) {
        lastSyncId = args.sync_id;
        if (typeof args.h_min === "number") hMin = args.h_min;
        if (typeof args.h_max === "number") hMax = args.h_max;
        draw();
      }
      sendMessage("streamlit:setFrameHeight", { height: 270 });
    }
  });

  draw();
  sendMessage("streamlit:componentReady", { apiVersion: 1 });
  sendMessage("streamlit:setFrameHeight", { height: 270 });
</script>
</body>
</html>
"""


@_cache_resource()
def get_hue_wheel_component():
    # 安全な専用一時ディレクトリ（パーミッション0700）を作成
    comp_dir = tempfile.mkdtemp(prefix="st_hue_wheel_")
    os.chmod(comp_dir, 0o700)
    index_path = os.path.join(comp_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(HUE_WHEEL_HTML)
    return components.declare_component("hue_wheel_v2", path=comp_dir)


hue_wheel_component = get_hue_wheel_component()


# ------------------------------------------------------------
# セキュリティ・ファイルI/Oユーティリティ
# ------------------------------------------------------------
def sanitize_filename(filename: str, default_stem: str = "image") -> str:
    base = os.path.basename(filename or "")
    stem = os.path.splitext(base)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._-")
    return cleaned[:80] if cleaned else default_stem


def sanitize_csv_cell(val):
    """CSV数式インジェクション対策"""
    if isinstance(val, str) and val.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + val
    return val


@_cache_data(show_spinner=False)
def download_url_bytes(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_URL_HOSTS:
        raise ValueError("許可されていないURLです。")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("ダウンロードサイズが上限を超えています。")
    return data


@_cache_data(show_spinner=False)
def load_repo_or_url_bytes(filename: str, raw_url: str) -> bytes:
    safe_name = os.path.basename(filename)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.abspath(os.path.join(app_dir, safe_name))
    if os.path.commonpath([app_dir, local_path]) == app_dir and os.path.isfile(local_path):
        if os.path.getsize(local_path) > MAX_DOWNLOAD_BYTES:
            raise ValueError("ファイルサイズが上限を超えています。")
        with open(local_path, "rb") as f:
            return f.read()
    return download_url_bytes(raw_url)


@_cache_data(show_spinner=False)
def load_image_rgb(file_bytes: bytes, filename: str) -> np.ndarray:
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("ファイルサイズが上限(100MB)を超えています。")

    ext = os.path.splitext(os.path.basename(filename))[1].lower()
    if ext == ".cr2":
        with tempfile.NamedTemporaryFile(suffix=".cr2", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rawpy.imread(tmp_path) as raw:
                return raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    with Image.open(io.BytesIO(file_bytes)) as raw_image:
        transposed = ImageOps.exif_transpose(raw_image)
        if transposed.mode in ["I", "I;16", "F"]:
            np_img = np.array(transposed)
            if np_img.max() > np_img.min():
                np_img = ((np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255.0).astype(np.uint8)
            else:
                np_img = np.zeros_like(np_img, dtype=np.uint8)
            return np.array(Image.fromarray(np_img).convert("RGB"), dtype=np.uint8)
        return np.array(transposed.convert("RGB"), dtype=np.uint8)


# ------------------------------------------------------------
# 画像処理・マスク演算関数
# ------------------------------------------------------------
@_cache_data(show_spinner=False)
def downscale_for_work(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_side:
        return img_rgb
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


@_cache_data(show_spinner=False)
def compute_superpixels(
    img_rgb: np.ndarray,
    n_segments: int,
    compactness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )
    boundaries = find_boundaries(segments, mode="inner")
    return segments.astype(np.int32), boundaries


def create_hsv_mask(
    img_rgb: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if h_min <= h_max:
        h_mask = (h >= h_min) & (h <= h_max)
    else:
        h_mask = (h >= h_min) | (h <= h_max)

    sv_mask = (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)
    return h_mask & sv_mask


def render_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    boundaries: Optional[np.ndarray],
    mask_color: Tuple[int, int, int] = (255, 60, 0),
    mask_alpha: float = 0.45,
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
    show_boundaries: bool = True,
) -> np.ndarray:
    canvas = img_rgb.astype(np.float32).copy()
    if np.any(mask):
        color_arr = np.array(mask_color, dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - mask_alpha) + color_arr * mask_alpha
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    if show_boundaries and boundaries is not None:
        out[boundaries] = boundary_color
    return out


def resize_for_display(img: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(
        img,
        (max_width, new_h),
        interpolation=cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA,
    )
    return resized, scale


def encode_png(img_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


def freedraw_objs_to_mask(
    objs: list,
    img_w: int,
    img_h: int,
    scale_x: float,
    scale_y: float,
    default_brush: int,
) -> np.ndarray:
    """st_canvasのjson_dataから安全に座標を検証してboolマスクを生成"""
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)

    for obj in objs[:500]:
        if not isinstance(obj, dict):
            continue
        try:
            raw_sw = float(obj.get("strokeWidth", default_brush))
            if not math.isfinite(raw_sw):
                raw_sw = float(default_brush)
        except (TypeError, ValueError):
            raw_sw = float(default_brush)

        stroke_w = max(1, min(300, int(round(raw_sw * scale_x))))
        r = max(1, stroke_w // 2)
        pts: List[Tuple[int, int]] = []

        path_data = obj.get("path")
        if isinstance(path_data, list):
            for cmd in path_data[:10000]:
                if isinstance(cmd, (list, tuple)):
                    nums = []
                    for v in cmd[1:]:
                        try:
                            fv = float(v)
                            if math.isfinite(fv):
                                nums.append(fv)
                        except (TypeError, ValueError):
                            pass
                    for j in range(0, len(nums) - 1, 2):
                        px = int(np.clip(round(nums[j] * scale_x), 0, img_w - 1))
                        py = int(np.clip(round(nums[j + 1] * scale_y), 0, img_h - 1))
                        pts.append((px, py))
        elif isinstance(path_data, str):
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", path_data)[:20000]]
            for j in range(0, len(nums) - 1, 2):
                px = int(np.clip(round(nums[j] * scale_x), 0, img_w - 1))
                py = int(np.clip(round(nums[j + 1] * scale_y), 0, img_h - 1))
                pts.append((px, py))

        if len(pts) >= 2:
            draw.line(pts, fill=255, width=stroke_w)
            for px, py in pts:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=255)
        elif len(pts) == 1:
            px, py = pts[0]
            draw.ellipse([px - r, py - r, px + r, py + r], fill=255)
        elif "left" in obj and "top" in obj:
            try:
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                w = max(1.0, float(obj.get("width", 10)) * float(obj.get("scaleX", 1.0)))
                h = max(1.0, float(obj.get("height", 10)) * float(obj.get("scaleY", 1.0)))
                if all(math.isfinite(v) for v in (left, top, w, h)):
                    x1 = int(np.clip(round(left * scale_x), 0, img_w - 1))
                    y1 = int(np.clip(round(top * scale_y), 0, img_h - 1))
                    x2 = int(np.clip(round((left + w) * scale_x), 0, img_w - 1))
                    y2 = int(np.clip(round((top + h) * scale_y), 0, img_h - 1))
                    draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255)
            except (TypeError, ValueError):
                pass

    return np.array(mask_img, dtype=np.uint8) > 0


def find_non_overlapping_label_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cx_pos: int,
    cy_pos: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
    mask_all: np.ndarray,
    placed_boxes: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """株領域および既配置ラベルと重ならない最適なラベル矩形を探索"""
    margins = [8, 20, 36, 56, 80]
    candidates = []

    for m in margins:
        raw_positions = [
            (cx_pos - box_w // 2, y1 - box_h - m),
            (cx_pos - box_w // 2, y2 + m),
            (x1 - box_w - m, cy_pos - box_h // 2),
            (x2 + m, cy_pos - box_h // 2),
            (x1 - box_w - m, y1 - box_h - m),
            (x2 + m, y1 - box_h - m),
            (x1 - box_w - m, y2 + m),
            (x2 + m, y2 + m),
        ]
        for lx, ly in raw_positions:
            lx1 = max(2, min(img_w - box_w - 2, lx))
            ly1 = max(2, min(img_h - box_h - 2, ly))
            lx2 = lx1 + box_w
            ly2 = ly1 + box_h

            label_overlap = 0
            for px1, py1, px2, py2 in placed_boxes:
                ox = max(0, min(lx2, px2) - max(lx1, px1))
                oy = max(0, min(ly2, py2) - max(ly1, py1))
                label_overlap += ox * oy

            mask_overlap = int(np.sum(mask_all[ly1:ly2, lx1:lx2]))
            lcx = (lx1 + lx2) / 2.0
            lcy = (ly1 + ly2) / 2.0
            dist = math.hypot(lcx - cx_pos, lcy - cy_pos)

            score = (mask_overlap * 1000.0) + (label_overlap * 500.0) + dist
            candidates.append((score, lx1, ly1, lx2, ly2))

    candidates.sort(key=lambda c: c[0])
    _, best_x1, best_y1, best_x2, best_y2 = candidates[0]
    return best_x1, best_y1, best_x2, best_y2


def init_state():
    defaults = {
        "file_key": None,
        "work_mask": None,
        "last_click": None,
        "last_preset": "イチゴNDVI用",
        "color_preset_select": "イチゴNDVI用",
        "h_min": 139,
        "h_max": 173,
        "s_range": (70, 255),
        "v_range": (50, 255),
        "wheel_sync_id": 0,
        "last_wheel_val": None,
        "sp_canvas_ver": 0,
        "sp_draw_objs": [],
        "last_image_source": None,
        "sp_n_segments": 250,
        "sp_compactness": 10.0,
        "sp_show_boundaries": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ------------------------------------------------------------
# 各タブのUI描画関数
# ------------------------------------------------------------
def render_tab1_hsv(img_rgb: np.ndarray, file_key: str) -> Tuple[int, int, int]:
    img_h, img_w = img_rgb.shape[:2]
    col1_ctrl, col1_view = st.columns([1, 2])

    with col1_ctrl:
        st.subheader("設定")
        preset = st.selectbox("色プリセット", list(COLOR_PRESETS.keys()), key="color_preset_select")

        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            (h0, h1), s_def, v_def = COLOR_PRESETS[preset]
            st.session_state.h_min = h0
            st.session_state.h_max = h1
            st.session_state.s_range = s_def
            st.session_state.v_range = v_def
            st.session_state.wheel_sync_id += 1

        mask_color_name = st.selectbox("マスクの色", list(MASK_DISPLAY_COLORS.keys()))
        selected_mask_color = MASK_DISPLAY_COLORS[mask_color_name]

        st.markdown("**H (色相) 範囲**（色相環のハンドルまたは範囲をドラッグ）")
        wheel_val = hue_wheel_component(
            h_min=st.session_state.h_min,
            h_max=st.session_state.h_max,
            sync_id=st.session_state.wheel_sync_id,
            key="hue_wheel_widget",
        )
        if (
            isinstance(wheel_val, dict)
            and "h_min" in wheel_val
            and "h_max" in wheel_val
            and wheel_val != st.session_state.last_wheel_val
        ):
            st.session_state.last_wheel_val = wheel_val
            st.session_state.h_min = int(np.clip(int(wheel_val["h_min"]), 0, 179))
            st.session_state.h_max = int(np.clip(int(wheel_val["h_max"]), 0, 179))

        s_min, s_max = st.slider("S (彩度) 範囲", 0, 255, key="s_range")
        v_min, v_max = st.slider("V (明度) 範囲", 0, 255, key="v_range")

        hsv_mask = create_hsv_mask(
            img_rgb,
            st.session_state.h_min,
            st.session_state.h_max,
            s_min,
            s_max,
            v_min,
            v_max,
        )

        if st.session_state.file_key != file_key:
            st.session_state.file_key = file_key
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.last_click = None
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
        elif st.session_state.work_mask is None or st.session_state.work_mask.shape != (img_h, img_w):
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.last_click = None

    with col1_view:
        st.subheader("プレビュー")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            hsv_preview = render_overlay(
                img_rgb,
                hsv_mask,
                None,
                mask_color=selected_mask_color,
                mask_alpha=0.5,
                show_boundaries=False,
            )
            st.image(hsv_preview, caption=f"元画像 + HSVマスク（{mask_color_name}）", use_container_width=True)
        with prev_col2:
            st.image((hsv_mask.astype(np.uint8)) * 255, caption="HSVマスク（白黒）", use_container_width=True)

        if st.button("このHSVマスクを②に反映", type="primary", use_container_width=True):
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.success("HSVマスクを「②スーパーピクセルで修正」に反映しました。")

    return selected_mask_color


def render_tab2_superpixel(
    img_rgb: np.ndarray,
    selected_mask_color: Tuple[int, int, int],
    binary_mask_u8: np.ndarray,
    safe_base_name: str,
) -> None:
    img_h, img_w = img_rgb.shape[:2]
    col2_ctrl, col2_main, col2_mask = st.columns([1, 1.8, 1.8])
    apply_brush_clicked = False

    with col2_ctrl:
        st.subheader("設定")
        edit_method = st.selectbox(
            "修正方法",
            ["スーパーピクセルをクリック（反転）", "直接描いて修正（ブラシ）"],
        )

        show_boundaries = st.checkbox("スーパーピクセル境界線を表示", key="sp_show_boundaries")
        st.divider()
        if edit_method == "スーパーピクセルをクリック（反転）":
            st.markdown("**スーパーピクセルの操作**")
            n_segments = st.slider(
                "スーパーピクセルの粒度（分割数）",
                min_value=50,
                max_value=1500,
                step=10,
                key="sp_n_segments",
                help="数値が大きいほどスーパーピクセルが細かくなります。",
            )
            compactness = st.slider(
                "コンパクトネス（形状の均一さ）",
                min_value=1.0,
                max_value=40.0,
                step=1.0,
                key="sp_compactness",
            )
            brush_radius = st.slider(
                "まとめてピクセルを選択",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="0にするとクリックした1つのスーパーピクセルのみを反転します。",
            )
        else:
            n_segments = st.session_state.sp_n_segments
            compactness = st.session_state.sp_compactness

            st.markdown("**ブラシの操作**")
            brush_action = st.radio("ブラシ操作", ["追加", "消去"], horizontal=True)
            brush_size = st.slider("ブラシの大きさ (px)", min_value=3, max_value=60, value=15, step=1)
            snap_to_sp = st.checkbox("描いた部分をスーパーピクセル単位に合わせる", value=False)
            if st.button("描画レイヤーをクリア", use_container_width=True):
                st.session_state.sp_canvas_ver += 1
                st.session_state.sp_draw_objs = []
                st.rerun()
            apply_brush_clicked = st.button("編集画面に適用", type="primary", use_container_width=True)

        st.divider()
        st.download_button(
            label="マスク画像をダウンロード (PNG)",
            data=encode_png(binary_mask_u8),
            file_name=f"{safe_base_name}_mask.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

    with st.spinner("スーパーピクセルを計算中..."):
        segments, boundaries = compute_superpixels(img_rgb, n_segments, compactness)

    with col2_main:
        st.subheader("編集画面")
        overlay_img = render_overlay(
            img_rgb,
            st.session_state.work_mask,
            boundaries,
            mask_color=selected_mask_color,
            show_boundaries=show_boundaries,
        )
        disp_img, _ = resize_for_display(overlay_img, BRUSH_DISPLAY_WIDTH)
        disp_h, disp_w = disp_img.shape[:2]

        if edit_method == "直接描いて修正（ブラシ）":
            stroke_col = "#00FF88" if brush_action == "追加" else "#FF3333"
            sp_draw_res = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color=stroke_col,
                background_image=Image.fromarray(disp_img),
                update_streamlit=True,
                drawing_mode="freedraw",
                width=disp_w,
                height=disp_h,
                key=f"sp_draw_canvas_{st.session_state.sp_canvas_ver}",
            )

            if sp_draw_res.json_data is not None and "objects" in sp_draw_res.json_data:
                st.session_state.sp_draw_objs = sp_draw_res.json_data["objects"]

            if apply_brush_clicked:
                objs = st.session_state.get("sp_draw_objs", [])
                if len(objs) > 0:
                    changed = False
                    for obj in objs:
                        stroke_work = freedraw_objs_to_mask(
                            [obj],
                            img_w,
                            img_h,
                            img_w / float(disp_w),
                            img_h / float(disp_h),
                            brush_size,
                        )
                        if not np.any(stroke_work):
                            continue
                        if snap_to_sp:
                            touched_labels = np.unique(segments[stroke_work])
                            target_region = np.isin(segments, touched_labels)
                        else:
                            target_region = stroke_work

                        obj_col = str(obj.get("stroke", "")).upper()
                        if obj_col == "#00FF88":
                            st.session_state.work_mask[target_region] = True
                        elif obj_col == "#FF3333":
                            st.session_state.work_mask[target_region] = False
                        else:
                            st.session_state.work_mask[target_region] = (brush_action == "追加")
                        changed = True

                    if changed:
                        st.session_state.sp_canvas_ver += 1
                        st.session_state.sp_draw_objs = []
                        st.rerun()
                    else:
                        st.warning("描画領域が取得できませんでした。もう一度描画してお試しください。")
                else:
                    st.warning("編集画面にブラシで描画してから「編集画面に適用」を押してください。")
        else:
            coord_kwargs = {"key": "sp_canvas"}
            if "use_column_width" in inspect.signature(streamlit_image_coordinates).parameters:
                coord_kwargs["use_column_width"] = "always"

            coords = streamlit_image_coordinates(Image.fromarray(disp_img), **coord_kwargs)
            if coords is not None:
                click_sig = (coords["x"], coords["y"], brush_radius, n_segments)
                if st.session_state.last_click != click_sig:
                    st.session_state.last_click = click_sig

                    rendered_w = max(1, int(coords.get("width", disp_w)))
                    rendered_h = max(1, int(coords.get("height", disp_h)))
                    orig_x = int(np.clip(round(coords["x"] * img_w / rendered_w), 0, img_w - 1))
                    orig_y = int(np.clip(round(coords["y"] * img_h / rendered_h), 0, img_h - 1))

                    if brush_radius == 0:
                        target_region = segments == segments[orig_y, orig_x]
                    else:
                        yy, xx = np.ogrid[:img_h, :img_w]
                        circle = (xx - orig_x) ** 2 + (yy - orig_y) ** 2 <= brush_radius ** 2
                        target_labels = np.unique(segments[circle])
                        target_region = np.isin(segments, target_labels)

                    center_val = st.session_state.work_mask[orig_y, orig_x]
                    st.session_state.work_mask[target_region] = not center_val
                    st.rerun()

    with col2_mask:
        st.subheader("マスク画像（白黒）")
        binary_mask_preview = (st.session_state.work_mask.astype(np.uint8)) * 255
        st.image(binary_mask_preview, use_container_width=True)


def render_tab3_ndvi(
    img_full_rgb: np.ndarray,
    export_mask: np.ndarray,
    using_sample: bool,
) -> None:
    st.subheader("株の範囲指定・NDVI解析")

    up_col1, up_col2 = st.columns(2)
    with up_col1:
        ndvi_uploaded = st.file_uploader(
            "NDVI画像をアップロード (.png / .jpg / .jpeg / .tif / .tiff)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="ndvi_uploader",
            help="未アップロードの場合は上部で選択した画像を使用します。",
        )
        if ndvi_uploaded is not None:
            try:
                ndvi_full_rgb = load_image_rgb(ndvi_uploaded.getvalue(), ndvi_uploaded.name)
            except Exception:
                st.error("NDVI画像の読み込みに失敗しました。ファイル形式やサイズをご確認ください。")
                return
        elif using_sample:
            try:
                with st.spinner("サンプルNDVI画像を読み込んでいます..."):
                    ndvi_bytes = load_repo_or_url_bytes(SAMPLE_NDVI_NAME, SAMPLE_NDVI_RAW_URL)
                    ndvi_full_rgb = load_image_rgb(ndvi_bytes, SAMPLE_NDVI_NAME)
                st.caption(f"セット済み: `{SAMPLE_NDVI_NAME}`")
            except Exception:
                st.warning("サンプルNDVI画像の読み込みに失敗したため元画像を使用します。")
                ndvi_full_rgb = img_full_rgb
        else:
            ndvi_full_rgb = img_full_rgb

    ndvi_h, ndvi_w = ndvi_full_rgb.shape[:2]

    with up_col2:
        mask_uploaded = st.file_uploader(
            "マスク画像をアップロード (白黒 .png / .jpg / .tif)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="mask_uploader",
            help="未アップロードの場合はタブ①・②で作成したマスクを使用します。",
        )
        if mask_uploaded is not None:
            try:
                mask_rgb = load_image_rgb(mask_uploaded.getvalue(), mask_uploaded.name)
                mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
                if mask_gray.shape != (ndvi_h, ndvi_w):
                    mask_gray = cv2.resize(mask_gray, (ndvi_w, ndvi_h), interpolation=cv2.INTER_NEAREST)
                analysis_mask = mask_gray > (127 if mask_gray.max() > 1 else 0)
            except Exception:
                st.error("マスク画像の読み込みに失敗しました。ファイル形式やサイズをご確認ください。")
                return
        elif using_sample:
            try:
                with st.spinner("サンプルマスク画像を読み込んでいます..."):
                    smask_bytes = load_repo_or_url_bytes(SAMPLE_MASK_NAME, SAMPLE_MASK_RAW_URL)
                    smask_rgb = load_image_rgb(smask_bytes, SAMPLE_MASK_NAME)
                    smask_gray = cv2.cvtColor(smask_rgb, cv2.COLOR_RGB2GRAY)
                    if smask_gray.shape != (ndvi_h, ndvi_w):
                        smask_gray = cv2.resize(smask_gray, (ndvi_w, ndvi_h), interpolation=cv2.INTER_NEAREST)
                    analysis_mask = smask_gray > (127 if smask_gray.max() > 1 else 0)
                st.caption(f"セット済み: `{SAMPLE_MASK_NAME}`")
            except Exception:
                st.warning("サンプルマスク画像の読み込みに失敗したため作業マスクを使用します。")
                analysis_mask = export_mask
        else:
            if export_mask.shape != (ndvi_h, ndvi_w):
                analysis_mask = cv2.resize(
                    export_mask.astype(np.uint8),
                    (ndvi_w, ndvi_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                analysis_mask = export_mask

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        default_min_area = 5000 if using_sample else 500
        min_plant_area = st.slider(
            "株として検出する最小面積 (px)",
            min_value=50,
            max_value=50000,
            value=default_min_area,
            step=50,
            key=f"min_plant_area_{using_sample}",
        )
    with col_s2:
        erode_size = st.slider(
            "隣接株の分離（収縮カーネルサイズ）",
            min_value=0,
            max_value=21,
            value=0,
            step=2,
            help="株同士がくっついている場合に大きくすると分離しやすくなります。",
        )

    seg_input = analysis_mask.astype(np.uint8) * 255
    if erode_size >= 3:
        kernel = np.ones((erode_size, erode_size), np.uint8)
        seg_input = cv2.morphologyEx(seg_input, cv2.MORPH_OPEN, kernel)

    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(seg_input)
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 165, 0),
        (255, 0, 255),
        (255, 255, 0),
        (100, 255, 180),
    ]

    raw_components = []
    heights = []
    for lbl in range(1, num_labels):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_plant_area:
            pmask = (labels_map == lbl) & analysis_mask
            ys, xs = np.where(pmask)
            if len(xs) == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            cx_pos = int(np.mean(xs))
            cy_pos = int(np.mean(ys))
            heights.append(y2 - y1)
            raw_components.append((cy_pos, cx_pos, pmask))

    if raw_components:
        row_tol = max(40, int(np.median(heights) * 0.6))
        raw_components.sort(key=lambda item: (item[0] // row_tol, item[1]))

    plant_masks = []
    for idx, (_, _, pmask) in enumerate(raw_components, start=1):
        crop_id = f"Strawberry_{idx:03d}"
        plant_masks.append((crop_id, pmask, palette[(idx - 1) % len(palette)]))

    has_ndvi_source = (ndvi_uploaded is not None) or using_sample
    ch0 = ndvi_full_rgb[:, :, 0].astype(np.int16)
    ch1 = ndvi_full_rgb[:, :, 1].astype(np.int16)
    ch2 = ndvi_full_rgb[:, :, 2].astype(np.int16)
    is_bw_ndvi = has_ndvi_source and (
        int(np.max(np.abs(ch0 - ch1))) <= 15 and int(np.max(np.abs(ch1 - ch2))) <= 15
    )

    st.divider()
    st.subheader("抽出・解析結果 ＆ ZIPダウンロード")

    annotated_arr = ndvi_full_rgb.copy()
    gray_full_ndvi = np.array(Image.fromarray(ndvi_full_rgb).convert("L"), dtype=np.float32)

    results = []
    placed_label_boxes: List[Tuple[int, int, int, int]] = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for crop_id, pmask, color_rgb in plant_masks:
            ys, xs = np.where(pmask)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1

            annotated_arr[pmask] = (
                annotated_arr[pmask].astype(np.float32) * 0.6
                + np.array(color_rgb, dtype=np.float32) * 0.4
            ).astype(np.uint8)

            contours, _ = cv2.findContours(pmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_arr, contours, -1, color_rgb, max(2, ndvi_w // 400))

            cx_pos = int(np.mean(xs))
            cy_pos = int(np.mean(ys))

            font_scale = max(0.55, ndvi_w / 1200.0)
            thickness = max(2, int(round(font_scale * 2)))
            (tw, th), baseline = cv2.getTextSize(crop_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pad = 6
            box_w = tw + pad * 2
            box_h = th + baseline + pad * 2

            lx1, ly1, lx2, ly2 = find_non_overlapping_label_box(
                x1, y1, x2, y2,
                cx_pos, cy_pos,
                box_w, box_h,
                ndvi_w, ndvi_h,
                analysis_mask,
                placed_label_boxes,
            )
            placed_label_boxes.append((lx1, ly1, lx2, ly2))

            lcx = (lx1 + lx2) // 2
            lcy = (ly1 + ly2) // 2
            cv2.line(annotated_arr, (lcx, lcy), (cx_pos, cy_pos), color_rgb, max(1, thickness - 1), cv2.LINE_AA)

            cv2.rectangle(annotated_arr, (lx1, ly1), (lx2, ly2), (20, 20, 20), -1)
            cv2.rectangle(annotated_arr, (lx1, ly1), (lx2, ly2), color_rgb, max(1, thickness - 1))
            cv2.putText(
                annotated_arr,
                crop_id,
                (lx1 + pad, ly1 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            masked_full = np.zeros_like(ndvi_full_rgb)
            masked_full[pmask] = ndvi_full_rgb[pmask]
            cropped_img = Image.fromarray(masked_full[y1:y2, x1:x2])

            crop_byte_arr = io.BytesIO()
            cropped_img.save(crop_byte_arr, format="PNG")
            zip_file.writestr(f"{crop_id}.png", crop_byte_arr.getvalue())

            plant_pixels = int(np.sum(pmask))
            mean_ndvi_idx = round(float(np.mean(gray_full_ndvi[pmask] / 255.0)), 3) if is_bw_ndvi else None

            results.append({
                "株ID": sanitize_csv_cell(crop_id),
                "株面積(px)": plant_pixels,
                "平均NDVI指標": mean_ndvi_idx,
            })

        annotated_byte_arr = io.BytesIO()
        Image.fromarray(annotated_arr).save(annotated_byte_arr, format="PNG")
        zip_file.writestr("annotated_full_image.png", annotated_byte_arr.getvalue())

        if len(results) > 0:
            df_csv = pd.DataFrame(results)
            zip_file.writestr("strawberry_analysis_results.csv", df_csv.to_csv(index=False).encode("utf-8-sig"))

    valid_crop_count = len(results)
    st.image(annotated_arr, caption=f"株セグメンテーション結果（検出数: {valid_crop_count} 株）", use_container_width=True)

    if valid_crop_count > 0:
        st.success(f"✅ 検出・指定された株の数: **{valid_crop_count} 株**")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("株がまだ検出・指定されていません。マスクを作成/アップロードしてください。")

    zip_buffer.seek(0)
    st.download_button(
        label=f"📦 全体画像・CSV・個別セグメンテーション画像（{valid_crop_count}枚）をZIPでダウンロード",
        data=zip_buffer,
        file_name="strawberry_ndvi_analysis_results.zip",
        mime="application/zip",
        type="primary",
        key="download_zip_btn",
        use_container_width=True,
    )


# ------------------------------------------------------------
# メインエントリポイント
# ------------------------------------------------------------
def main():
    init_state()

    st.title("NDVI解析 & セグメンテーションアプリ")
    st.caption(
        "①HSVと②スーパーピクセルでマスク画像を作成・ダウンロードし、③でNDVI画像とマスク画像を用いた株ごとのセグメンテーション解析が行えます。"
    )

    image_source = st.radio(
        "画像の選択方法",
        [
            "ファイルをアップロード",
            f"サンプル画像 ({SAMPLE_SEG_NAME}) を使用",
        ],
        horizontal=True,
    )

    using_sample = image_source != "ファイルをアップロード"
    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None

    if not using_sample:
        uploaded_file = st.file_uploader(
            "画像をアップロード (.png / .jpg / .jpeg / .tif / .tiff / .heic / .cr2)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "cr2"],
        )
        if uploaded_file is None:
            st.info("上のアップローダーから画像ファイルを選択するか、サンプル画像を選んでください。")
            st.stop()
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
    else:
        try:
            with st.spinner("サンプル画像を読み込んでいます..."):
                file_bytes = load_repo_or_url_bytes(SAMPLE_SEG_NAME, SAMPLE_SEG_URL)
                file_name = SAMPLE_SEG_NAME
        except Exception:
            st.error("サンプル画像の読み込みに失敗しました。")
            st.stop()

    file_key = f"{os.path.basename(file_name)}_{len(file_bytes)}"

    if st.session_state.last_image_source != image_source:
        st.session_state.last_image_source = image_source
        if using_sample:
            st.session_state.color_preset_select = "イチゴNDVI用"
            st.session_state.last_preset = "イチゴNDVI用"
            st.session_state.h_min = 139
            st.session_state.h_max = 173
            st.session_state.s_range = (70, 255)
            st.session_state.v_range = (50, 255)
            st.session_state.wheel_sync_id += 1

    try:
        with st.spinner("画像を読み込んでいます..."):
            img_full_rgb = load_image_rgb(file_bytes, file_name)
    except Exception:
        st.error("画像の読み込みに失敗しました。ファイル形式やサイズをご確認ください。")
        st.stop()

    orig_h, orig_w = img_full_rgb.shape[:2]
    img_rgb = downscale_for_work(img_full_rgb, WORK_MAX_SIDE) if max(orig_h, orig_w) > WORK_MAX_SIDE else img_full_rgb
    img_h, img_w = img_rgb.shape[:2]

    tab1, tab2, tab3 = st.tabs([
        "① HSVでマスク作成",
        "②スーパーピクセルで修正",
        "③ 株の範囲指定・NDVI解析",
    ])

    with tab1:
        selected_mask_color = render_tab1_hsv(img_rgb, file_key)

    if (img_h, img_w) != (orig_h, orig_w):
        export_mask = cv2.resize(
            st.session_state.work_mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        export_mask = st.session_state.work_mask

    binary_mask_u8 = (export_mask.astype(np.uint8)) * 255
    safe_base_name = sanitize_filename(file_name)

    with tab2:
        render_tab2_superpixel(img_rgb, selected_mask_color, binary_mask_u8, safe_base_name)

    with tab3:
        render_tab3_ndvi(img_full_rgb, export_mask, using_sample)


if __name__ == "__main__":
    main()
