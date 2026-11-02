import io
import zipfile
import numpy as np
import pandas as P
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ページの設定
st.set_page_config(
    page_title="イチゴ株検出・NDVI解析アプリ", page_icon="🍓", layout="wide"
)

st.title("🍓 イチゴの株検出・NDVI解析アプリ")
st.markdown(
    "イチゴの畝を真上から撮影し、**NDVI値を抽出して二値化した画像**をアップロードしてください。"
    "画面上で四角形を描画して株を指定し、手動で位置や大きさを補正できます。"
)

# サイドバー設定
st.sidebar.header("⚙️ 設定・ツール")
stroke_color = st.sidebar.color_picker("枠線の色", "#00FF00")  # デフォルト：緑
stroke_width = st.sidebar.slider("枠線の太さ", 1, 5, 2)

# 画像アップロード
uploaded_file = st.file_uploader(
    "二値化されたNDVI画像をアップロード", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  # 画像の読み込み
  image = Image.open(uploaded_file).convert("RGB")
  img_width, img_height = image.size

  st.sidebar.success(f"画像読込成功 ({img_width} × {img_height} px)")

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

  # Drawable Canvas の配置
  canvas_result = st_canvas(
      fill_color="rgba(0, 255, 0, 0.2)",  # 半透明の塗りつぶし
      stroke_width=stroke_width,
      stroke_color=stroke_color,
      background_image=image,
      update_streamlit=True,
      drawing_mode="rect",
      point_radius=2,
      width=canvas_width,
      height=canvas_height,
      key="canvas_strawberry",
  )

  # 描画された四角形（オブジェクト）の処理
  if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj["type"] == "rect"]

    if len(rects) > 0:
      st.divider()
      st.subheader("2. 抽出・解析結果")
      st.write(f"検出・指定された株の数: **{len(rects)} 株**")

      # キャンバス座標から元画像座標への変換スケール
      scale_x = img_width / canvas_width
      scale_y = img_height / canvas_height

      results = []
      zip_buffer = io.BytesIO()

      # ZIPファイルの作成準備
      with zipfile.ZipFile(
          zip_buffer, "w", zipfile.ZIP_DEFLATED
      ) as zip_file:
        for i, rect in enumerate(rects):
          # 座標とサイズ（回転やスケーリングを考慮）
          rx = rect["left"] * scale_x
          ry = rect["top"] * scale_y
          rw = rect["width"] * rect["scaleX"] * scale_x
          rh = rect["height"] * rect["scaleY"] * scale_y

          # 整数化し、画像範囲内に収める
          x1 = max(0, int(rx))
          y1 = max(0, int(ry))
          x2 = min(img_width, int(rx + rw))
          y2 = min(img_height, int(ry + rh))

          if x2 > x1 and y2 > y1:
            # 画像の切り出し (Crop)
            cropped_img = image.crop((x1, y1, x2, y2))

            # 囲んだ部分のNDVI指標計算（二値化画像をグレースケール変換して処理）
            crop_gray = cropped_img.convert("L")
            crop_np = np.array(crop_gray)

            # 二値化画像（0 or 255）を前提とした植生ピクセル割合の算出
            if crop_np.max() > 1:
              binary_crop = crop_np > 127
            else:
              binary_crop = crop_np > 0

            total_pixels = binary_crop.size
            vegetation_pixels = np.sum(binary_crop)
            veg_ratio = (
                (vegetation_pixels / total_pixels) * 100
                if total_pixels > 0
                else 0
            )
            mean_intensity = (
                np.mean(crop_np) / 255.0
            )  # 0.0〜1.0の範囲での平均値（NDVI指標の目安）

            # 結果リストに追加
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

            # 個別画像をメモリ上のZIPに追加
            img_byte_arr = io.BytesIO()
            cropped_img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            zip_file.writestr(
                f"strawberry_{i+1:03d}_x{x1}_y{y1}.png", img_byte_arr.read()
            )

      # 表形式で結果を表示
      df = P.DataFrame(results)
      st.dataframe(df, use_container_width=True)

      # 3. ZIP一括ダウンロードボタン
      st.divider()
      st.subheader("3. 画像の一括ダウンロード")
      zip_buffer.seek(0)
      st.download_button(
          label="📦 切り抜いた株画像をZIPでダウンロード",
          data=zip_buffer,
          file_name="strawberry_cropped_images.zip",
          mime="application/zip",
      )
    else:
      st.info(
          "👆 画像上で「矩形ツール」を選択し、解析したい株の範囲を四角形で囲んでください。"
      )
else:
  st.info(
      "📂 アプリケーションを開始するには、二値化されたNDVI画像をアップロードしてください。"
  )

# フッター
st.markdown("---")
st.markdown(
    "**💡 使い方と仕様メモ**\n"
    "- **手動補正**: 描画した枠をマウスでクリックすると、四隅のハンドルでサイズ変更や位置の微調整が可能です。\n"
    "- **NDVI値計算**: 二値化された画像内の「白ピクセルの割合（植生率）」や「平均輝度値」を算出して指標化しています。\n"
)
