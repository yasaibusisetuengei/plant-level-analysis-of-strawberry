import io
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
    "（PNG, JPG, TIF/TIFF形式）をアップロードしてください。"
)

# 画像アップロード（.tif, .tiffファイルに対応）
uploaded_file = st.file_uploader(
    "二値化されたNDVI画像をアップロード", type=["png", "jpg", "jpeg", "tif", "tiff"]
)


def hex_to_rgb(hex_code):
  hex_code = hex_code.lstrip("#")
  return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


if uploaded_file is not None:
  try:
    # 画像の読み込み
    raw_image = Image.open(uploaded_file)

    # 16bitや浮動小数点型などの特殊なTIFを標準的な8bit RGBに変換
    if raw_image.mode in ["I", "I;16", "F"]:
      np_img = np.array(raw_image)
      if np_img.max() > np_img.min():
        np_img = (
            (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255
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
        "- ツールバーの **「矩形（Rectangle）ツール」** を選択し、画像上で株を囲んでください。\n"
        "- 描画後、四角形をクリックして**位置の移動、拡大・縮小、削除**による手動補正が可能です。"
    )

    # 画面幅に合わせてキャンバスの表示サイズを調整（最大幅 750px）
    max_canvas_width = 750
    if img_width > max_canvas_width:
      canvas_width = max_canvas_width
      canvas_height = int(img_height * (max_canvas_width / img_width))
    else:
      canvas_width = img_width
      canvas_height = img_height

    # 画面下のアコーディオンによる設定（枠線の色・太さ）
    with st.expander("⚙️ 描画設定（枠線の色や太さを変更する場合）"):
      col1, col2 = st.columns(2)
      with col1:
        stroke_color = st.color_picker("枠線の色", "#00FF00")  # デフォルト：緑
      with col2:
        stroke_width = st.slider("枠線の太さ", 1, 5, 2)

    # セッションステートの初期化（描画データの維持用）
    if "rects_data" not in st.session_state:
      st.session_state["rects_data"] = []

    # Drawable Canvas の配置
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.2)",
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

    # 全体画像コピー作成
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    rgb_color = hex_to_rgb(stroke_color)

    # フォントの読み込み（デフォルト）
    try:
      font = ImageFont.load_default()
    except Exception:
      font = None

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

          # ① 全体画像に枠を描画
          draw.rectangle(
              [x1, y1, x2, y2], outline=rgb_color, width=stroke_width
          )

          # ② 全体画像の枠の近くに株ID（テキスト）を描画
          text_x = x1
          text_y = max(0, y1 - 15)  # 枠の少し上に配置
          draw.text((text_x, text_y), crop_id, fill=rgb_color, font=font)

          # ③ 該当部分を切り取り
          cropped_img = image.crop((x1, y1, x2, y2))

          # NDVI計算（二値化画像評価）
          crop_gray = cropped_img.convert("L")
          crop_np = np.array(crop_gray)

          binary_crop = crop_np > 127 if crop_np.max() > 1 else crop_np > 0
          total_pixels = binary_crop.size
          vegetation_pixels = np.sum(binary_crop)
          veg_ratio = (
              (vegetation_pixels / total_pixels) * 100 if total_pixels > 0 else 0
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

      # ④ 枠線とIDを描画した全体の高解像度画像をZIPに保存
      annotated_byte_arr = io.BytesIO()
      annotated_img.save(annotated_byte_arr, format="PNG")
      annotated_byte_arr.seek(0)
      zip_file.writestr("annotated_full_image.png", annotated_byte_arr.read())

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
    else:
      st.info(
          "👆 画像上で株を四角形で囲むと、自動的に解析が行われここに表が表示されます。"
      )

    # ダウンロードボタン
    st.markdown(
        "📥 **ダウンロードZIPに含まれる内容**:\n"
        "- `annotated_full_image.png` （四角形の枠と株IDが書き込まれた全体画像）\n"
        "- `strawberry_analysis_results.csv` （株ごとの解析結果データ）\n"
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
else:
  st.info(
      "📂 アプリケーションを開始するには、二値化されたNDVI画像（PNG, JPG, TIF等）をアップロードしてください。"
  )
