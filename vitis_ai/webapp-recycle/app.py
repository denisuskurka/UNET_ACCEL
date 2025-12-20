import os

# File: Utils
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import time
import shutil
import numpy as np
from flask import Flask, request, url_for, send_from_directory
from PIL import Image

# Utils
from util import crop_image
from rects_from_masks import find_best_rectangle, create_binary_rectangle_mask
from crop_rect_from_original import scale_mask_to_original, get_mask_bounding_box, crop_image_by_box
from paste_ellipse_back import paste_ellipse_back 

# Model Imports
# We alias them to switch between them easily
from infer_stem import infer_stem
from infer_ellipse import infer_ellipse as infer_ellipse_unet
from infer_ellipse_regress import infer_ellipse as infer_ellipse_reg

import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = './data'
FIG_FOLDER = './fig'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)

IMG_SIZE = (256, 256)
INPUT_BMP = "data.bmp"
CROPPED_PNG = "data_cropped.png"
STEM_INPUT_PNG = "data_stem_input.png"
STEM_INPUT_BIN = "data_stem_input.bin"
RESULT_BIN = "result.bin"
RESULT_PNG = "data_result.png"
RECT_MASK_BMP = "data_rectangle_mask.bmp"
FINAL_CROPPED_PNG = "data_cropped_final.png"
PAINTED_IN_CROPPED = "painted_in_cropped.png"

@app.route('/data/<filename>')
def uploaded_file(filename):
    """Serve files from the upload folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/fig/<filename>')
def serve_figure(filename):
    """Serve files from the fig folder for logos/static assets."""
    return send_from_directory(FIG_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Variables for displaying images on the page
    uploaded_file_url = None
    result_image_url = None
    rectangle_mask_url = None
    final_cropped_url = None
    ellipse_inference_url = None
    painted_in_cropped_url = None
    filename = None
    
    # Defaults for the form
    selected_infer_mode = "sw"
    selected_ellipse_mode = "unet"

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
        # 3) Get User Options
        # ------------------------------------------------------------------
        # Mode: SW (TensorFlow) or HW (DPU)
        selected_infer_mode = request.form.get("infer_mode", "sw")
        is_hw = (selected_infer_mode == "hw")

        # Ellipse Model: U-Net (Segmentation) or Regressor (Parameters)
        selected_ellipse_mode = request.form.get("ellipse_mode", "unet")

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
        print(f"[INFO] Running STEM Inference (Mode: {selected_infer_mode.upper()})")
        arr_out = infer_stem(cropped_filepath, hw=is_hw)
        
        if arr_out is None:
            return "infer_stem returned no result."

        # Reshape and Normalize for display
        arr_out = arr_out.reshape(IMG_SIZE)
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

                # Ensure filename matches what the inference scripts expect
                final_cropped_const = os.path.join(UPLOAD_FOLDER, "data_cropped_final.png")
                # (The save above might have already saved it there, but strict check:)
                if os.path.abspath(final_cropped_path) != os.path.abspath(final_cropped_const):
                    shutil.copy(final_cropped_path, final_cropped_const)

                # --------------------------------------------------------------
                # 7a) Ellipse Inference (Selection Logic)
                # --------------------------------------------------------------
                print(f"[INFO] Running Ellipse Inference using: {selected_ellipse_mode.upper()} (Mode: {selected_infer_mode.upper()})")
                
                if selected_ellipse_mode == "unet":
                    # Use Segmentation U-Net
                    infer_ellipse_unet(hw=is_hw)
                else:
                    # Use Parameter Regressor
                    infer_ellipse_reg(hw=is_hw)

                # Result is saved to "ellipse_infer.png" by both scripts
                ellipse_infer_png = os.path.join(UPLOAD_FOLDER, "ellipse_infer.png")
                if os.path.exists(ellipse_infer_png):
                    ellipse_inference_url = url_for('uploaded_file', filename="ellipse_infer.png")

                # --------------------------------------------------------------
                # 7b) Paste the final (with ellipse) back into the bigger cropped image
                # --------------------------------------------------------------
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
    # Helper for checked attributes
    sw_checked = "checked" if selected_infer_mode == "sw" else ""
    hw_checked = "checked" if selected_infer_mode == "hw" else ""
    unet_checked = "checked" if selected_ellipse_mode == "unet" else ""
    reg_checked = "checked" if selected_ellipse_mode == "regress" else ""

    return f'''
    <html>
      <head>
        <title>Vitis AI Inference App</title>
        <style>
            body {{ margin:20px; font-family:sans-serif; position: relative; }}
            .control-group {{ margin-bottom: 15px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: fit-content; }}
            .control-group h3 {{ margin-top: 0; font-size: 16px; color: #555; }}
            label {{ margin-right: 15px; cursor: pointer; }}
            input[type=submit] {{ padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            input[type=submit]:hover {{ background-color: #0056b3; }}
            
            /* Logo Container Styling */
            .logo-container {{
                position: absolute;
                top: 10px;
                right: 20px;
                display: flex;
                gap: 20px;
                align-items: center;
                background: white;
                padding: 10px;
                border-radius: 8px;
            }}
            .logo-container img {{
                height: 60px; /* Adjust height as needed */
                width: auto;
            }}
        </style>
      </head>
      <body>
        
        <div class="logo-container">
            <img src="./fig/evropa_logo.npng" alt="European Union Logo">
            <img src="./fig/npo_logo.npng" alt="Narodni plan obnovy Logo">
        </div>

        <h1>Vitis AI Inference Pipeline</h1>

        <form method="POST" action="/" enctype="multipart/form-data">
          
          <div style="display:flex; gap: 20px;">
              <div class="control-group">
                <h3>1. Execution Mode</h3>
                <label>
                    <input type="radio" name="infer_mode" value="sw" {sw_checked} /> 
                    Software (TensorFlow .h5)
                </label>
                <label>
                    <input type="radio" name="infer_mode" value="hw" {hw_checked} /> 
                    Hardware (DPU .xmodel)
                </label>
              </div>

              <div class="control-group">
                <h3>2. Ellipse Strategy</h3>
                <label>
                    <input type="radio" name="ellipse_mode" value="unet" {unet_checked} /> 
                    Segmentation (U-Net)
                </label>
                <label>
                    <input type="radio" name="ellipse_mode" value="regress" {reg_checked} /> 
                    Regression (Parameters)
                </label>
              </div>
          </div>

          <br/>
          <div class="control-group">
            <h3>3. Upload Input</h3>
            <input type="file" name="file" accept=".bmp" required />
            <br/><br/>
            <input type="submit" value="Run Inference"/>
          </div>
        </form>

        <hr/>
        
        <div style="display: flex; flex-wrap: wrap; gap: 40px; margin-top:20px;">
          <div>
            <h2>Input (Cropped)</h2>
            {f'<img src="{uploaded_file_url}" style="max-width:400px;border:1px solid #ccc;"/>' 
              if uploaded_file_url else '<p style="color:#888;">Waiting for upload...</p>'}
          </div>
          <div>
            <h2>Final Result</h2>
            {f'<img src="{painted_in_cropped_url}" style="max-width:400px;border:2px solid #28a745;"/>' 
              if painted_in_cropped_url else '<p style="color:#888;">Result will appear here.</p>'}
          </div>
        </div>

        <hr style="margin:40px 0;"/>

        <h2>Pipeline Debug View</h2>
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:20px;">
          <div style="text-align:center;">
             <p><b>Step 1:</b> STEM Output</p>
             {f'<img src="{result_image_url}" style="max-width:200px; border:1px solid #ccc;" />' if result_image_url else '-'}
          </div>
          <div style="text-align:center;">
             <p><b>Step 2:</b> ROI Mask</p>
             {f'<img src="{rectangle_mask_url}" style="max-width:200px; border:1px solid #ccc;" />' if rectangle_mask_url else '-'}
          </div>
          <div style="text-align:center;">
             <p><b>Step 3:</b> Final ROI Crop</p>
             {f'<img src="{final_cropped_url}" style="max-width:200px; border:1px solid #ccc;" />' if final_cropped_url else '-'}
          </div>
          <div style="text-align:center;">
             <p><b>Step 4:</b> Ellipse Inference</p>
             {f'<img src="{ellipse_inference_url}" style="max-width:200px; border:1px solid #ccc;" />' if ellipse_inference_url else '-'}
          </div>
        </div>
      </body>
    </html>
    '''

if __name__ == '__main__':
    # Important: debug=False for VART/DPU stability
    app.run(debug=False, host="0.0.0.0", port=5000)

