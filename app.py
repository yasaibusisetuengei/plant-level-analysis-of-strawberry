import io
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ページの設定
st.set_page_config(
    page_title="イチゴ株検出・NDVI解析アプリ", page_icon="🍓", layout="wide"
)

st.title("🍓 イチゴの株検出・NDVI解析アプリ")
st.markdown(
    "イチゴの畝を真上から撮影し、**NDVI値を抽出して二値化した画像**"
    "（PNG, JPG, TIF/TIFF形式）をアップロードするか、サンプル画像でお試しください。"
)


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def parse_color_to_rgb(color_val, default_rgb):
    """canvasのオブジェクトから取得した色（HEXやRGB文字列）をRGBタプルに変換"""
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


# サンプル画像のダウンロード用関数
@st.cache_data
def get_sample_image():
    url = "https://github.com/yasaibusisetuengei/plant-level-analysis-of-strawberry/releases/download/sample/IMG_3714.tif"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        image_bytes = response.read()
    return io.BytesIO(image_bytes)


# 条件①：サンプル画像を選択できるようにする
use_sample = st.checkbox(
    "🧪 サンプル画像を使用して試す", value=False
)

if use_sample:
    try:
        uploaded_file = get_sample_image()
        st.info("💡 サンプル画像を読み込みました。")
    except Exception as e:
        st.error(f"サンプル画像のダウンロードに失敗しました: {e}")
        uploaded_file = None
else:
    uploaded_file = st.file_uploader(
        "二値化されたNDVI画像をアップロード",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

if uploaded_file is not None:
    try:
        # 画像の読み込み
        raw_image = Image.open(uploaded_file)

        # 16bitや浮動小数点型などの特殊なTIFを標準的な8bit RGBに変換
        if raw_image.mode in ["I", "I;16", "F"]:
            np_img = np.array(raw_image)
            if np_img.max() > np_img.min():
                np_img = (
                    (np_img - np_img.min())
                    / (np_img.max() - np_img.min())
                    * 255
                ).astype(np.uint8)
            else:
                np_img = np.zeros_like(np_img, dtype=np.uint8)
            image = Image.fromarray(np_img).convert("RGB")
        else:
            image = raw_image.convert("RGB")

        img_width, img_height = image.size
        st.success(f"画像読込成功 ({img_width} × {img_height} px)")

        st.subheader("1. 株の範囲を指定する（手動補正可能）")
        st.markdown(
            "- マウスでドラックし、画像上で株を囲んでください。\n"
            "- 描画後、画像の右上に表示されるアイコンから**元に戻す、修正、全印の削除**が可能です。"
        )

        # 画面幅に合わせてキャンバスの表示サイズを調整（最大幅 750px）
        max_canvas_width = 750
        if img_width > max_canvas_width:
            canvas_width = max_canvas_width
            canvas_height = int(img_height * (max_canvas_width / img_width))
        else:
            canvas_width = img_width
            canvas_height = img_height

        # 画面下のアコーディオンによる設定（枠線・塗りつぶしの色、太さ）
        with st.expander(
            "⚙️ 描画設定（枠線の色や太さを変更する場合）", expanded=True
        ):
            col1, col2 = st.columns(2)
            with col1:
                stroke_color = st.color_picker(
                    "枠線・塗りつぶしの色", "#00FF00", key="stroke_color_picker"
                )
            with col2:
                stroke_width = st.slider(
                    "枠線の太さ", 1, 5, 2, key="stroke_width_slider"
                )

        # 選択された色をRGBに変換し、キャンバス用の半透明RGBAを作成
        rgb_color = hex_to_rgb(stroke_color)
        fill_color_rgba = (
            f"rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)"
        )

        # セッションステートの初期化（描画データの維持用）
        if "rects_data" not in st.session_state:
            st.session_state["rects_data"] = []

        # Drawable Canvas の配置
        canvas_result = st_canvas(
            fill_color=fill_color_rgba,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=image,
            update_streamlit=True,
            drawing_mode="rect",
            width=canvas_width,
            height=canvas_height,
            return_image_data=False,
            key="canvas_strawberry",
        )

        # キャンバスからデータをリアルタイムに取得してセッションステートに蓄積
        if (
            canvas_result.json_data is not None
            and "objects" in canvas_result.json_data
        ):
            objs = canvas_result.json_data["objects"]
            valid_objs = [
                obj
                for obj in objs
                if obj.get("type") == "rect"
                or ("width" in obj and "height" in obj)
            ]
            st.session_state["rects_data"] = valid_objs

        # -------------------------------------------------------------------------
        # 解析結果およびダウンロードセクション
        # -------------------------------------------------------------------------
        st.divider()
        st.subheader("2. 抽出・解析結果 ＆ ダウンロード")

        rects = st.session_state.get("rects_data", [])

        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height

        results = []
        zip_buffer = io.BytesIO()

        # 全体画像コピー作成 (RGBAにして半透明合成の準備)
        annotated_img = image.convert("RGBA")
        overlay = Image.new("RGBA", annotated_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            valid_crop_count = 0

            for i, rect in enumerate(rects):
                left = rect.get("left", 0)
                top = rect.get("top", 0)
                width = rect.get("width", 0) * rect.get("scaleX", 1.0)
                height = rect.get("height", 0) * rect.get("scaleY", 1.0)

                x_start = left
                y_start = top
                x_end = left + width
                y_end = top + height

                x1 = max(0, int(round(min(x_start, x_end) * scale_x)))
                y1 = max(0, int(round(min(y_start, y_end) * scale_y)))
                x2 = min(img_width, int(round(max(x_start, x_end) * scale_x)))
                y2 = min(img_height, int(round(max(y_start, y_end) * scale_y)))

                # わずかな誤タップを除外するため、幅・高さが一定以上あるものだけ処理
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    valid_crop_count += 1
                    crop_id = f"Strawberry_{valid_crop_count:03d}"
                    crop_name = f"{crop_id}.png"

                    # 条件②：オブジェクトごとの描画色（途中で変更した色も含む）を正確に取得
                    obj_stroke = rect.get("stroke", stroke_color)
                    rect_rgb = parse_color_to_rgb(obj_stroke, rgb_color)

                    # ① オーバーレイに半透明の塗りつぶしと個別設定色の枠線を描画
                    overlay_draw.rectangle(
                        [x1, y1, x2, y2],
                        fill=rect_rgb + (int(255 * 0.2),),
                        outline=rect_rgb + (255,),
                        width=stroke_width,
                    )

                    # ② 株IDを四角形の真ん中（中心座標）に10倍の大きさで描画
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    txt_w, txt_h = len(crop_id) * 6, 11
                    txt_img = Image.new(
                        "RGBA", (txt_w, txt_h), (0, 0, 0, 0)
                    )
                    txt_draw = ImageDraw.Draw(txt_img)
                    txt_draw.text((0, 0), crop_id, fill=rect_rgb + (255,))

                    scaled_txt = txt_img.resize(
                        (txt_w * 10, txt_h * 10), Image.Resampling.NEAREST
                    )
                    scaled_w, scaled_h = scaled_txt.size

                    text_x = center_x - scaled_w // 2
                    text_y = center_y - scaled_h // 2

                    overlay.paste(scaled_txt, (text_x, text_y), scaled_txt)

                    # ③ 該当部分を切り取り（元画像から）
                    cropped_img = image.crop((x1, y1, x2, y2))

                    # NDVI計算（二値化画像評価）
                    crop_gray = cropped_img.convert("L")
                    crop_np = np.array(crop_gray)

                    binary_crop = (
                        crop_np > 127 if crop_np.max() > 1 else crop_np > 0
                    )
                    total_pixels = binary_crop.size
                    vegetation_pixels = np.sum(binary_crop)
                    veg_ratio = (
                        (vegetation_pixels / total_pixels) * 100
                        if total_pixels > 0
                        else 0
                    )
                    mean_intensity = np.mean(crop_np) / 255.0

                    results.append({
                        "株ID": crop_id,
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                        "幅 (px)": x2 - x1,
                        "高さ (px)": y2 - y1,
                        "植生ピクセル割合 (%)": round(veg_ratio, 2),
                        "平均NDVI指標": round(mean_intensity, 3),
                    })

                    # 切り抜いた個別の株画像をZIPに保存
                    crop_byte_arr = io.BytesIO()
                    cropped_img.save(crop_byte_arr, format="PNG")
                    crop_byte_arr.seek(0)
                    zip_file.writestr(crop_name, crop_byte_arr.read())

            # ④ オーバーレイを元画像と合成してRGBに変換し、ZIPに保存
            final_annotated_img = Image.alpha_composite(
                annotated_img, overlay
            ).convert("RGB")
            annotated_byte_arr = io.BytesIO()
            final_annotated_img.save(annotated_byte_arr, format="PNG")
            annotated_byte_arr.seek(0)
            zip_file.writestr(
                "annotated_full_image.png", annotated_byte_arr.read()
            )

            # ⑤ 解析結果の表をCSVファイルとしてZIPに保存
            if len(results) > 0:
                df_csv = pd.DataFrame(results)
                csv_data = df_csv.to_csv(index=False).encode("utf-8-sig")
                zip_file.writestr("strawberry_analysis_results.csv", csv_data)

        # テーブル表示
        if valid_crop_count > 0:
            st.success(f"✅ 検出・指定された株の数: **{valid_crop_count} 株**")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

        # ダウンロードボタン
        st.markdown(
            "📥 **ダウンロードZIPに含まれる内容**:\n"
            "- `annotated_full_image.png` （設定色に連動した枠線・半透明塗りつぶし・株の真ん中に5倍サイズの株ID入り全体画像）\n"
            "- `strawberry_analysis_results.csv` （株ごとの解析結果CSVデータ）\n"
            f"- `Strawberry_001.png` 〜 （各四角形で切り抜いた個別画像：計 **{valid_crop_count}** 枚）"
        )

        zip_buffer.seek(0)
        st.download_button(
            label=(
                f"📦 全体画像・CSV・個別画像（{valid_crop_count}枚）をZIPでダウンロード"
            ),
            data=zip_buffer,
            file_name="strawberry_ndvi_analysis_results.zip",
            mime="application/zip",
            key="download_zip_btn",
        )

    except Exception as e:
        st.error(
            f"画像の読み込みまたは処理中にエラーが発生しました（ファイル形式をご確認ください）: {e}"
        )
