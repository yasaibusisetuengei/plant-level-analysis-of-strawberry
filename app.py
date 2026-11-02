import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
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

    # Drawable Canvas の配置
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.2)",  # 半透明の塗りつぶし
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=image,
        update_streamlit=True,
        drawing_mode="rect",
        width=canvas_width,
        height=canvas_height,
        return_image_data=True,
        key="canvas_strawberry",
    )

    # -------------------------------------------------------------------------
    # 解析結果およびダウンロードセクション（常時表示）
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("2. 抽出・解析結果 ＆ ダウンロード")

    rects = []
    if canvas_result.json_data is not None:
      objects = canvas_result.json_data.get("objects", [])
      rects = [obj for obj in objects if obj["type"] == "rect"]

    # ZIPバッファとデータフレームの準備
    results = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
      # ① 印（四角形の枠）をつけた全体の画像をZIPに追加
      if canvas_result.image_data is not None:
        canvas_img_array = canvas_result.image_data.astype(np.uint8)
        annotated_img = Image.fromarray(canvas_img_array, mode="RGBA").convert(
            "RGB"
        )
        annotated_img_resized = annotated_img.resize(
            (img_width, img_height), Image.Resampling.LANCZOS
        )

        annotated_byte_arr = io.BytesIO()
        annotated_img_resized.save(annotated_byte_arr, format="PNG")
        annotated_byte_arr.seek(0)
        zip_file.writestr(
            "annotated_full_image.png", annotated_byte_arr.read()
        )

      # キャンバス座標から元画像座標への変換スケール
      scale_x = img_width / canvas_width
      scale_y = img_height / canvas_height

      # ② 囲んだ部分を個別に切り取ってZIPに追加
      for i, rect in enumerate(rects):
        rx = rect["left"] * scale_x
        ry = rect["top"] * scale_y
        rw = rect["width"] * rect["scaleX"] * scale_x
        rh = rect["height"] * rect["scaleY"] * scale_y

        x1 = max(0, int(rx))
        y1 = max(0, int(ry))
        x2 = min(img_width, int(rx + rw))
        y2 = min(img_height, int(ry + rh))

        if x2 > x1 and y2 > y1:
          cropped_img = image.crop((x1, y1, x2, y2))

          crop_gray = cropped_img.convert("L")
          crop_np = np.array(crop_gray)

          if crop_np.max() > 1:
            binary_crop = crop_np > 127
          else:
            binary_crop = crop_np > 0

          total_pixels = binary_crop.size
          vegetation_pixels = np.sum(binary_crop)
          veg_ratio = (
              (vegetation_pixels / total_pixels) * 100 if total_pixels > 0 else 0
          )
          mean_intensity = np.mean(crop_np) / 255.0

          results.append({
              "株ID": f"Strawberry_{i+1:03d}",
              "X1": x1,
              "Y1": y1,
              "X2": x2,
              "Y2": y2,
              "幅 (px)": x2 - x1,
              "高さ (px)": y2 - y1,
              "植生ピクセル割合 (%)": round(veg_ratio, 2),
              "平均NDVI指標": round(mean_intensity, 3),
          })

          img_byte_arr = io.BytesIO()
          cropped_img.save(img_byte_arr, format="PNG")
          img_byte_arr.seek(0)
          zip_file.writestr(
              f"cropped_images/strawberry_{i+1:03d}_x{x1}_y{y1}.png",
              img_byte_arr.read(),
          )

    # 解析結果の表表示（株がある場合のみ）
    if len(rects) > 0:
      st.write(f"検出・指定された株の数: **{len(rects)} 株**")
      df = pd.DataFrame(results)
      st.dataframe(df, use_container_width=True)
    else:
      st.info(
          "👆 画像上で株を四角形で囲むと、ここに株ごとの解析結果（表）がリアルタイムに表示されます。"
      )

    # 常にダウンロードボタンを表示
    st.markdown(
        "📥 **ダウンロード内容**:\n"
        "- `annotated_full_image.png` （四角形の枠をつけた全体の画像）\n"
        "- `cropped_images/` フォルダ （切り抜かれた個別の株画像群 ※株を囲むと自動追加されます）"
    )
    zip_buffer.seek(0)
    st.download_button(
        label=(
            "📦 印付き全体画像 ＆ 切り抜き画像をZIPで一括ダウンロード"
            f" ({len(rects)}株選択中)"
        ),
        data=zip_buffer,
        file_name="strawberry_ndvi_analysis_results.zip",
        mime="application/zip",
    )

  except Exception as e:
    st.error(
        f"画像の読み込みまたは処理中にエラーが発生しました（ファイル形式をご確認ください）: {e}"
    )
else:
  st.info(
      "📂 アプリケーションを開始するには、二値化されたNDVI画像（PNG, JPG, TIF等）をアップロードしてください。"
  )
