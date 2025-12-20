import os

# File: app.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import time
import shutil
import numpy as np
from flask import Flask, request, url_for, send_from_directory
from PIL import Image

from util import crop_image
from rects_from_masks import find_best_rectangle, create_binary_rectangle_mask
from crop_rect_from_original import scale_mask_to_original, get_mask_bounding_box, crop_image_by_box
from infer_ellipse import infer_ellipse
from infer_stem import infer_stem
from fit_ellipse import fit_ellipse
from paint_ellipse import paint_ellipse
from paste_ellipse_back import paste_ellipse_back 
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = './data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (128, 128)
INPUT_BMP = "data.bmp"
CROPPED_PNG = "data_cropped.png"
STEM_INPUT_PNG = "data_stem_input.png"
STEM_INPUT_BIN = "data_stem_input.bin"
RESULT_BIN = "result.bin"
RESULT_PNG = "data_result.png"
RECT_MASK_BMP = "data_rectangle_mask.bmp"
FINAL_CROPPED_PNG = "data_cropped_final.png"
PAINTED_IN_CROPPED = "painted_in_cropped.png"

ELLIPSE_INFER_BMP = "ellipse_infer.bmp"  # infer_ellipse script writes here by default

@app.route('/data/<filename>')
def uploaded_file(filename):
    """Serve files from the upload folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Variables for displaying images on the page
    uploaded_file_url = None
    result_image_url = None
    rectangle_mask_url = None
    final_cropped_url = None
    ellipse_inference_url = None
    ellipse_fitted_url = None
    ellipse_final_url = None
    painted_in_cropped_url = None
    filename = None

    if request.method == 'POST':
        # ------------------------------------------------------------------
        # 1) Clean the ./data folder
        # ------------------------------------------------------------------
        for f in os.listdir(UPLOAD_FOLDER):
            path_ = os.path.join(UPLOAD_FOLDER, f)
            try:
                if os.path.isfile(path_):
                    os.remove(path_)
                else:
                    shutil.rmtree(path_)
            except Exception as e:
                print(f"Could not remove {path_}. Reason: {e}")

        # ------------------------------------------------------------------
        # 2) Check uploaded file
        # ------------------------------------------------------------------
        if 'file' not in request.files:
            return "No file part in request."
        file = request.files['file']
        if file.filename == '':
            return "No selected file."

        filename = file.filename
        if not filename.lower().endswith('.bmp'):
            return "Error: Only BMP files are accepted."

        # ------------------------------------------------------------------
        # 3) Check if user selected HW or SW
        # ------------------------------------------------------------------
        infer_mode = request.form.get("infer_mode", "sw")  # 'sw' or 'hw'
        is_hw = (infer_mode == "hw")

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, INPUT_BMP)
        file.save(file_path)

        # ------------------------------------------------------------------
        # 4) Crop => data_cropped.png
        # ------------------------------------------------------------------
        cropped_filepath = os.path.join(UPLOAD_FOLDER, CROPPED_PNG)
        with Image.open(file_path) as img:
            img_cropped = crop_image(img)
            img_cropped.save(cropped_filepath)

        uploaded_file_url = url_for('uploaded_file', filename=CROPPED_PNG)

        # ------------------------------------------------------------------
        # 5) Run STEM inference (SW/HW)
        # ------------------------------------------------------------------
        arr_out = infer_stem(cropped_filepath, hw=is_hw)
        if arr_out is None:
            return "infer_stem returned no result."

        arr_out = arr_out.reshape(IMG_SIZE)

        # Convert [0..1] to [0..255] for display
        arr_out_clamped = np.clip(arr_out, 0.0, 1.0) * 255.0
        arr_out_clamped = arr_out_clamped.astype(np.uint8)

        result_png_filepath = os.path.join(UPLOAD_FOLDER, RESULT_PNG)
        result_img = Image.fromarray(arr_out_clamped, mode='L')
        result_img.save(result_png_filepath)
        result_image_url = url_for('uploaded_file', filename=RESULT_PNG)

        # ------------------------------------------------------------------
        # 6) Find best rectangle => data_rectangle_mask.bmp
        # ------------------------------------------------------------------
        rect_w, rect_h = 32, 32
        (best_x, best_y), best_sum = find_best_rectangle(
            arr_out_clamped, rect_w, rect_h
        )
        rect_mask = create_binary_rectangle_mask(
            IMG_SIZE[1],  # height=128
            IMG_SIZE[0],  # width=128
            best_x, best_y,
            rect_w, rect_h
        )

        rect_mask_bmp_path = os.path.join(UPLOAD_FOLDER, RECT_MASK_BMP)
        Image.fromarray(rect_mask).save(rect_mask_bmp_path)
        rectangle_mask_url = url_for('uploaded_file', filename=RECT_MASK_BMP)

        # ------------------------------------------------------------------
        # 7) Scale mask => bounding box => crop the image
        # ------------------------------------------------------------------
        with Image.open(cropped_filepath) as cropped_img:
            cw, ch = cropped_img.size

        mask_128_pil = Image.fromarray(rect_mask, mode='L')
        mask_scaled_pil = mask_128_pil.resize((cw, ch), Image.NEAREST)
        mask_scaled = np.array(mask_scaled_pil)

        bbox = get_mask_bounding_box(mask_scaled)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            with Image.open(cropped_filepath) as cropped_img:
                final_crop = crop_image_by_box(cropped_img, (xmin, ymin, xmax, ymax))
                final_cropped_path = os.path.join(UPLOAD_FOLDER, FINAL_CROPPED_PNG)
                final_crop.save(final_cropped_path)
                final_cropped_url = url_for('uploaded_file', filename=FINAL_CROPPED_PNG)

                # 7a) Ellipse inference
                final_cropped_const = os.path.join(UPLOAD_FOLDER, "data_cropped_final.png")
                if final_cropped_path != final_cropped_const:
                    os.replace(final_cropped_path, final_cropped_const)

                infer_ellipse()

                # ellipse_infer.png
                ellipse_infer_png = os.path.join(UPLOAD_FOLDER, "ellipse_infer.png")
                if os.path.exists(ellipse_infer_png):
                    ellipse_inference_url = url_for('uploaded_file', filename="ellipse_infer.png")

                # 7b) Paste the final (with ellipse) back into the bigger cropped image
                #     => painted_in_cropped.png
                painted_in_cropped_png = os.path.join(UPLOAD_FOLDER, PAINTED_IN_CROPPED)
                paste_ellipse_back(
                    big_image_path=cropped_filepath,
                    small_painted_path=ellipse_infer_png,
                    out_path=painted_in_cropped_png,
                    bbox=(xmin, ymin, xmax, ymax)
                )

                if os.path.exists(painted_in_cropped_png):
                    painted_in_cropped_url = url_for(
                        'uploaded_file', filename=PAINTED_IN_CROPPED
                    )

    # -----------------------
    # Construct final HTML
    # -----------------------
    return f'''
    <html>
      <head>
        <title>STEM Inference</title>
      </head>
      <body style="margin:20px;font-family:sans-serif;">
        <h1>STEM Model Inference</h1>

        <!-- 1) The same menu (SW/HW + file upload) -->
        <form method="POST" action="/" enctype="multipart/form-data">
          <label>
            <input type="radio" name="infer_mode" value="sw" checked />
            STEM UNET in SW
          </label>
          <label style="margin-left: 20px;">
            <input type="radio" name="infer_mode" value="hw" />
            STEM UNET in HW
          </label>
          <br/><br/>
          <input type="file" name="file" accept=".bmp"/>
          <input type="submit" value="Upload"/>
        </form>

        <!-- 2) The new two-column layout: left = original cropped, right = final painted -->
        <hr/>
        <div style="display: flex; flex-wrap: wrap; gap: 40px; margin-top:20px;">
          <div>
            <h2>Cropped Input</h2>
            {f'<img src="{uploaded_file_url}" style="max-width:400px;border:1px solid #ccc;"/>' 
              if uploaded_file_url else ''}
          </div>
          <div>
            <h2>Final Painted Ellipse</h2>
            {f'<img src="{painted_in_cropped_url}" style="max-width:400px;border:1px solid #ccc;"/>' 
              if painted_in_cropped_url else ''}
          </div>
        </div>

        <!-- 3) Separator line -->
        <hr style="margin:40px 0;"/>

        <!-- 4) Debug pipeline images -->
        <h2>Debug Pipeline</h2>
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:20px;">
          {f'<div><p>STEM Output</p><img src="{result_image_url}" style="max-width:200px; border:1px solid #ccc;" /></div>' 
            if result_image_url else ''}
          {f'<div><p>Rect Mask</p><img src="{rectangle_mask_url}" style="max-width:200px; border:1px solid #ccc;" /></div>' 
            if rectangle_mask_url else ''}
          {f'<div><p>Final Crop</p><img src="{final_cropped_url}" style="max-width:200px; border:1px solid #ccc;" /></div>' 
            if final_cropped_url else ''}
          {f'<div><p>Ellipse Inference</p><img src="{ellipse_inference_url}" style="max-width:200px; border:1px solid #ccc;" /></div>' 
            if ellipse_inference_url else ''}
        </div>
      </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

