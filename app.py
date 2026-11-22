import inspect
import io
import os
import re
import tempfile
import urllib.request
import zipfile
from typing import Optional, Tuple

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

# HEIC/HEIFサポートを有効化
pillow_heif.register_heif_opener()

st.set_page_config(
    page_title="イチゴ株検出・NDVI解析 & セグメンテーションアプリ",
    page_icon="🍓",
    layout="wide",
)

DISPLAY_WIDTH = 600
WORK_MAX_SIDE = 1024
SAMPLE_SEG_NAME = "IMG_3715.jpg"
SAMPLE_SEG_URL = "https://raw.githubusercontent.com/yasaibusisetuengei/segmentation-app/main/IMG_3715.jpg"
SAMPLE_NDVI_NAME = "IMG_3714.tif"
SAMPLE_NDVI_URL = "https://github.com/yasaibusisetuengei/plant-level-analysis-of-strawberry/releases/download/sample/IMG_3714.tif"

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


@st.cache_resource
def get_hue_wheel_component():
    comp_dir = os.path.join(tempfile.gettempdir(), "st_hue_wheel_component_v2")
    os.makedirs(comp_dir, exist_ok=True)
    index_path = os.path.join(comp_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(HUE_WHEEL_HTML)
    return components.declare_component("hue_wheel_v2", path=comp_dir)


hue_wheel_component = get_hue_wheel_component()


# ------------------------------------------------------------
# ユーティリティ関数
# ------------------------------------------------------------
def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def parse_color_to_rgb(color_val, default_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not color_val:
        return default_rgb
    if isinstance(color_val, str):
        if color_val.startswith("#"):
            try:
                return hex_to_rgb(color_val)
            except Exception:
                return default_rgb
        elif color_val.startswith("rgb"):
            nums = re.findall(r"[\d.]+", color_val)
            if len(nums) >= 3:
                return (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
    return default_rgb


@st.cache_data(show_spinner=False)
def download_url_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


@st.cache_data(show_spinner=False)
def load_image_rgb(file_bytes: bytes, filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".cr2":
        with tempfile.NamedTemporaryFile(suffix=".cr2", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rawpy.imread(tmp_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            return rgb
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    raw_image = Image.open(io.BytesIO(file_bytes))
    raw_image = ImageOps.exif_transpose(raw_image)

    # 16bitや浮動小数点型などの特殊なTIFを標準的な8bit RGBに変換
    if raw_image.mode in ["I", "I;16", "F"]:
        np_img = np.array(raw_image)
        if np_img.max() > np_img.min():
            np_img = (
                (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255.0
            ).astype(np.uint8)
        else:
            np_img = np.zeros_like(np_img, dtype=np.uint8)
        return np.array(Image.fromarray(np_img).convert("RGB"), dtype=np.uint8)

    return np.array(raw_image.convert("RGB"), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def downscale_for_work(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_side:
        return img_rgb
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


@st.cache_data(show_spinner=False)
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


def canvas_obj_to_mask(
    obj: dict,
    img_w: int,
    img_h: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """drawable_canvasの図形(rect / path / polygon)を元画像サイズのboolマスクに変換"""
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    obj_type = obj.get("type", "")

    if obj_type == "path" and "path" in obj:
        pts = []
        obj_left = obj.get("left", 0)
        obj_top = obj.get("top", 0)
        sx = obj.get("scaleX", 1.0)
        sy = obj.get("scaleY", 1.0)
        # path要素をポリゴンとして近似
        for cmd in obj["path"]:
            if len(cmd) >= 3 and isinstance(cmd[1], (int, float)) and isinstance(cmd[2], (int, float)):
                px = int(round(cmd[1] * scale_x))
                py = int(round(cmd[2] * scale_y))
                pts.append((px, py))
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
        elif len(pts) >= 2:
            draw.line(pts, fill=255, width=max(5, int(round(obj.get("strokeWidth", 10) * scale_x))))
    else:
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0) * obj.get("scaleX", 1.0)
        height = obj.get("height", 0) * obj.get("scaleY", 1.0)
        x1 = max(0, int(round(min(left, left + width) * scale_x)))
        y1 = max(0, int(round(min(top, top + height) * scale_y)))
        x2 = min(img_w, int(round(max(left, left + width) * scale_x)))
        y2 = min(img_h, int(round(max(top, top + height) * scale_y)))
        if (x2 - x1) > 2 and (y2 - y1) > 2:
            draw.rectangle([x1, y1, x2, y2], fill=255)

    return np.array(mask_img, dtype=np.uint8) > 0


# ------------------------------------------------------------
# 色プリセット & マスク表示色定義
# ------------------------------------------------------------
COLOR_PRESETS = {
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
# セッション状態初期化
# ------------------------------------------------------------
def init_state():
    defaults = {
        "file_key": None,
        "work_mask": None,
        "last_click": None,
        "last_preset": "緑",
        "h_min": 35,
        "h_max": 85,
        "s_range": (40, 255),
        "v_range": (40, 255),
        "wheel_sync_id": 0,
        "last_wheel_val": None,
        "ndvi_objs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ------------------------------------------------------------
# メインUI
# ------------------------------------------------------------
st.title("🍓 イチゴ株検出・NDVI解析 & セグメンテーションアプリ")
st.caption(
    "①HSVと②スーパーピクセルで植物マスクを作成・修正し、③ダウンロードおよび④マスク画像に従った株ごとのセグメンテーションNDVI解析が行えます。"
)

compress_for_work = st.checkbox(
    "作業中は画像を縮小・圧縮して高速化し、ダウンロード・解析時に元の解像度へ戻す",
    value=True,
    help="CR2などの高解像度画像で処理が重い場合に有効にしてください。長辺を最大1024pxに縮小して作業し、出力時に元画像のサイズへ復元します。",
)

image_source = st.radio(
    "画像の選択方法",
    [
        "ファイルをアップロード",
        f"サンプル画像 ({SAMPLE_SEG_NAME}) を使用",
        f"サンプルNDVI画像 ({SAMPLE_NDVI_NAME}) を使用",
    ],
    horizontal=True,
)

file_bytes: Optional[bytes] = None
file_name: Optional[str] = None

if image_source == "ファイルをアップロード":
    uploaded_file = st.file_uploader(
        "画像をアップロード (.png / .jpg / .jpeg / .tif / .tiff / .heic / .cr2)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "cr2"],
    )
    if uploaded_file is None:
        st.info("上のアップローダーから画像ファイルを選択するか、サンプル画像を選んでください。")
        st.stop()
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif image_source == f"サンプル画像 ({SAMPLE_SEG_NAME}) を使用":
    try:
        with st.spinner("サンプル画像を読み込んでいます..."):
            local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAMPLE_SEG_NAME)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    file_bytes = f.read()
            else:
                file_bytes = download_url_bytes(SAMPLE_SEG_URL)
            file_name = SAMPLE_SEG_NAME
    except Exception as e:
        st.error(f"サンプル画像の読み込みに失敗しました: {e}")
        st.stop()
else:
    try:
        with st.spinner("サンプルNDVI画像を読み込んでいます..."):
            file_bytes = download_url_bytes(SAMPLE_NDVI_URL)
            file_name = SAMPLE_NDVI_NAME
    except Exception as e:
        st.error(f"サンプルNDVI画像のダウンロードに失敗しました: {e}")
        st.stop()

file_key = f"{file_name}_{len(file_bytes)}"

with st.spinner("画像を読み込んでいます..."):
    img_full_rgb = load_image_rgb(file_bytes, file_name)

orig_h, orig_w = img_full_rgb.shape[:2]

if compress_for_work:
    img_rgb = downscale_for_work(img_full_rgb, WORK_MAX_SIDE)
else:
    img_rgb = img_full_rgb

img_h, img_w = img_rgb.shape[:2]

tab1, tab2, tab3, tab4 = st.tabs([
    "① HSVでマスク作成",
    "②スーパーピクセルで修正",
    "ダウンロード",
    "④ 株の範囲指定・NDVI解析",
])

# ------------------------------------------------------------
# タブ①: HSV値に従って大雑把に分ける
# ------------------------------------------------------------
with tab1:
    col1_ctrl, col1_view = st.columns([1, 2])

    with col1_ctrl:
        st.subheader("設定")
        preset = st.selectbox(
            "色プリセット",
            ["緑", "赤", "青", "黄", "白", "黒", "カスタム"],
        )

        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            if preset in COLOR_PRESETS:
                (h0, h1), s_def, v_def = COLOR_PRESETS[preset]
                st.session_state.h_min = h0
                st.session_state.h_max = h1
                st.session_state.s_range = s_def
                st.session_state.v_range = v_def
                st.session_state.wheel_sync_id += 1

        mask_color_name = st.selectbox(
            "マスクの色",
            list(MASK_DISPLAY_COLORS.keys()),
        )
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
            st.session_state.h_min = int(wheel_val["h_min"])
            st.session_state.h_max = int(wheel_val["h_max"])

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

        # 新しい画像が読み込まれたら初期マスクをセット
        if st.session_state.file_key != file_key:
            st.session_state.file_key = file_key
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.last_click = None
            st.session_state.ndvi_objs = []
        elif st.session_state.work_mask is None or st.session_state.work_mask.shape != (img_h, img_w):
            if st.session_state.work_mask is None:
                st.session_state.work_mask = hsv_mask.copy()
            else:
                st.session_state.work_mask = cv2.resize(
                    st.session_state.work_mask.astype(np.uint8),
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
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
            st.success("HSVマスクを「②スーパーピクセルで修正」に反映しました。")

# ------------------------------------------------------------
# タブ②: スーパーピクセルで修正
# ------------------------------------------------------------
with tab2:
    col2_ctrl, col2_main, col2_mask = st.columns([1, 1.8, 1.8])

    with col2_ctrl:
        st.subheader("設定")
        n_segments = st.slider(
            "スーパーピクセルの粒度（分割数）",
            min_value=50,
            max_value=1500,
            value=250,
            step=10,
            help="数値が大きいほどスーパーピクセルが細かくなります。",
        )
        compactness = st.slider(
            "コンパクトネス（形状の均一さ）",
            min_value=1.0,
            max_value=40.0,
            value=10.0,
            step=1.0,
        )
        show_boundaries = st.checkbox("スーパーピクセル境界線を表示", value=True)
        brush_radius = st.slider(
            "まとめてピクセルを選択",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="0にするとクリックした1つのスーパーピクセルのみを反転します。",
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
        disp_img, _ = resize_for_display(overlay_img, DISPLAY_WIDTH)
        disp_h, disp_w = disp_img.shape[:2]

        coord_kwargs = {"key": "sp_canvas"}
        if "use_column_width" in inspect.signature(streamlit_image_coordinates).parameters:
            coord_kwargs["use_column_width"] = "always"

        coords = streamlit_image_coordinates(
            Image.fromarray(disp_img),
            **coord_kwargs,
        )

        if coords is not None:
            click_sig = (coords["x"], coords["y"], brush_radius, n_segments)
            if st.session_state.last_click != click_sig:
                st.session_state.last_click = click_sig

                rendered_w = coords.get("width", disp_w)
                rendered_h = coords.get("height", disp_h)
                orig_x = int(np.clip(round(coords["x"] * img_w / rendered_w), 0, img_w - 1))
                orig_y = int(np.clip(round(coords["y"] * img_h / rendered_h), 0, img_h - 1))

                if brush_radius == 0:
                    target_label = segments[orig_y, orig_x]
                    target_region = segments == target_label
                else:
                    yy, xx = np.ogrid[:img_h, :img_w]
                    circle = (xx - orig_x) ** 2 + (yy - orig_y) ** 2 <= brush_radius ** 2
                    target_labels = np.unique(segments[circle])
                    target_region = np.isin(segments, target_labels)

                center_val = st.session_state.work_mask[orig_y, orig_x]
                st.session_state.work_mask[target_region] = not center_val

                st.rerun()

    with col2_mask:
        st.subheader("最終マスク画像（白黒）")
        binary_mask_preview = (st.session_state.work_mask.astype(np.uint8)) * 255
        st.image(binary_mask_preview, use_container_width=True)

# ------------------------------------------------------------
# タブ③: ダウンロード (元解像度へ復元)
# ------------------------------------------------------------
if (img_h, img_w) != (orig_h, orig_w):
    export_mask = cv2.resize(
        st.session_state.work_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
else:
    export_mask = st.session_state.work_mask

binary_mask_u8 = (export_mask.astype(np.uint8)) * 255
final_overlay = render_overlay(
    img_full_rgb,
    export_mask,
    None,
    mask_color=selected_mask_color,
    show_boundaries=False,
)

with tab3:
    st.subheader("ダウンロード")
    if (img_h, img_w) != (orig_h, orig_w):
        st.caption(f"作業解像度: {img_w}×{img_h} px → 出力解像度: {orig_w}×{orig_h} px（元サイズに復元済み）")
    else:
        st.caption(f"出力解像度: {orig_w}×{orig_h} px")

    mask_ratio = float(np.mean(export_mask) * 100.0)
    st.metric("マスク面積率", f"{mask_ratio:.2f}%")

    base_name = os.path.splitext(file_name)[0]
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.image(img_full_rgb, caption="元画像", use_container_width=True)
        st.download_button(
            label="元画像をダウンロード (PNG)",
            data=encode_png(img_full_rgb),
            file_name=f"{base_name}_original.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_col2:
        st.image(binary_mask_u8, caption="最終マスク画像（白黒）", use_container_width=True)
        st.download_button(
            label="マスク画像をダウンロード (PNG)",
            data=encode_png(binary_mask_u8),
            file_name=f"{base_name}_mask.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

    with dl_col3:
        st.image(final_overlay, caption="最終オーバーレイ画像", use_container_width=True)
        st.download_button(
            label="オーバーレイ画像をダウンロード (PNG)",
            data=encode_png(final_overlay),
            file_name=f"{base_name}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

# ------------------------------------------------------------
# タブ④: 株の範囲指定・NDVI解析 (マスク画像に基づくセグメンテーション)
# ------------------------------------------------------------
with tab4:
    st.subheader("株の範囲指定・NDVI解析（マスク画像連動セグメンテーション）")

    use_raw_as_mask = st.checkbox(
        "アップロード画像自体がすでに二値化NDVI画像の場合は、その白領域をマスクとして使用する",
        value=False,
    )
    if use_raw_as_mask:
        gray_full = cv2.cvtColor(img_full_rgb, cv2.COLOR_RGB2GRAY)
        analysis_mask = gray_full > (127 if gray_full.max() > 1 else 0)
    else:
        analysis_mask = export_mask

    seg_mode = st.radio(
        "株の指定方法",
        [
            "マスクの連結領域から自動セグメンテーション",
            "手動で囲んでマスク領域を切り出し（四角形 / フリーハンド）",
        ],
        horizontal=True,
    )

    plant_masks = []  # List of (crop_id, mask_bool, color_rgb)
    pil_full_img = Image.fromarray(img_full_rgb)

    if seg_mode == "マスクの連結領域から自動セグメンテーション":
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            min_plant_area = st.slider(
                "株として検出する最小面積 (px)",
                min_value=50,
                max_value=50000,
                value=500,
                step=50,
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
        idx = 0
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_plant_area:
                idx += 1
                crop_id = f"Strawberry_{idx:03d}"
                # 収縮後の領域を元マスクの範囲内で対応付け
                pmask = (labels_map == lbl) & analysis_mask
                plant_masks.append((crop_id, pmask, palette[(idx - 1) % len(palette)]))

    else:
        st.markdown(
            "- マウスでドラッグし、株を囲んでください。**囲んだ範囲とマスク画像の重なり部分のみ**が株としてセグメンテーション抽出されます。"
        )
        max_canvas_width = 750
        if orig_w > max_canvas_width:
            canvas_width = max_canvas_width
            canvas_height = int(orig_h * (max_canvas_width / orig_w))
        else:
            canvas_width = orig_w
            canvas_height = orig_h

        with st.expander("⚙️ 描画設定（描画モード・枠線の色や太さ）", expanded=True):
            c_draw1, c_draw2, c_draw3 = st.columns(3)
            with c_draw1:
                draw_mode = st.selectbox("描画モード", ["rect", "freedraw"], format_func=lambda x: "四角形 (rect)" if x == "rect" else "フリーハンド (freedraw)")
            with c_draw2:
                stroke_color = st.color_picker("枠線・塗りつぶしの色", "#00FF00", key="stroke_color_picker")
            with c_draw3:
                stroke_width = st.slider("枠線の太さ", 1, 5, 2, key="stroke_width_slider")

        rgb_color = hex_to_rgb(stroke_color)
        fill_color_rgba = f"rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)"

        # 背景にはマスクを薄く重ねた画像を表示し、株の位置を見やすくする
        canvas_bg = Image.fromarray(
            render_overlay(
                img_full_rgb,
                analysis_mask,
                None,
                mask_color=selected_mask_color,
                mask_alpha=0.35,
                show_boundaries=False,
            )
        )

        canvas_result = st_canvas(
            fill_color=fill_color_rgba,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=canvas_bg,
            update_streamlit=True,
            drawing_mode=draw_mode,
            width=canvas_width,
            height=canvas_height,
            key="canvas_strawberry",
        )

        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            st.session_state["ndvi_objs"] = canvas_result.json_data["objects"]

        scale_x = orig_w / canvas_width
        scale_y = orig_h / canvas_height

        for i, obj in enumerate(st.session_state.get("ndvi_objs", [])):
            shape_mask = canvas_obj_to_mask(obj, orig_w, orig_h, scale_x, scale_y)
            # マスク画像に従ってセグメンテーション的に切り出し
            pmask = shape_mask & analysis_mask
            if np.any(pmask):
                obj_stroke = obj.get("stroke", stroke_color)
                rect_rgb = parse_color_to_rgb(obj_stroke, rgb_color)
                plant_masks.append((f"Strawberry_{len(plant_masks) + 1:03d}", pmask, rect_rgb))

    # -------------------------------------------------------------------------
    # 解析結果の描画・集計・ZIP作成
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("抽出・解析結果 ＆ ZIPダウンロード")

    annotated_arr = img_full_rgb.copy()
    gray_full_ndvi = np.array(pil_full_img.convert("L"), dtype=np.float32)

    results = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for crop_id, pmask, color_rgb in plant_masks:
            ys, xs = np.where(pmask)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1

            # セグメンテーション領域を半透明で塗りつぶし＋輪郭線描画
            annotated_arr[pmask] = (
                annotated_arr[pmask].astype(np.float32) * 0.6
                + np.array(color_rgb, dtype=np.float32) * 0.4
            ).astype(np.uint8)

            contours, _ = cv2.findContours(pmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_arr, contours, -1, color_rgb, max(2, orig_w // 400))

            # 重心に株IDを描画
            cx_pos = int(np.mean(xs))
            cy_pos = int(np.mean(ys))
            font_scale = max(0.6, orig_w / 1000.0)
            thickness = max(2, int(round(font_scale * 2)))
            cv2.putText(
                annotated_arr,
                crop_id,
                (max(0, cx_pos - 60), cy_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_arr,
                crop_id,
                (max(0, cx_pos - 60), cy_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color_rgb,
                thickness,
                cv2.LINE_AA,
            )

            # セグメンテーション領域のみで個別株画像（背景黒抜き）を切り出し
            masked_full = np.zeros_like(img_full_rgb)
            masked_full[pmask] = img_full_rgb[pmask]
            cropped_arr = masked_full[y1:y2, x1:x2]
            cropped_img = Image.fromarray(cropped_arr)

            crop_byte_arr = io.BytesIO()
            cropped_img.save(crop_byte_arr, format="PNG")
            zip_file.writestr(f"{crop_id}.png", crop_byte_arr.getvalue())

            # セグメンテーション領域内のNDVI指標・面積集計
            plant_pixels = int(np.sum(pmask))
            bbox_area = (x2 - x1) * (y2 - y1)
            veg_ratio = (plant_pixels / bbox_area) * 100.0 if bbox_area > 0 else 0.0
            mean_ndvi_idx = float(np.mean(gray_full_ndvi[pmask]) / 255.0)

            results.append({
                "株ID": crop_id,
                "株面積 (px)": plant_pixels,
                "枠内植生割合 (%)": round(veg_ratio, 2),
                "平均NDVI指標": round(mean_ndvi_idx, 3),
                "重心X": cx_pos,
                "重心Y": cy_pos,
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
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
        st.info("株がまだ検出・指定されていません。マスクを作成するか、範囲を描画してください。")

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
