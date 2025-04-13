import os
import time
import shutil
import numpy as np
from flask import Flask, request, url_for, send_from_directory
from PIL import Image

from util import crop_image
from rects_from_masks import find_best_rectangle, create_binary_rectangle_mask
from crop_rect_from_original import scale_mask_to_original, get_mask_bounding_box, crop_image_by_box
# The below script references final_cropped image as INPUT_PIC and writes ellipse_infer.bmp as OUTPUT_EXPORT
from infer_ellipse import infer_ellipse
from infer_stem import infer_stem
from fit_ellipse import fit_ellipse
from paint_ellipse import paint_ellipse
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

ELLIPSE_INFER_BMP = "ellipse_infer.bmp"  # infer_ellipse script writes here by default

@app.route('/data/<filename>')
def uploaded_file(filename):
    """Serve files from the upload folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_file_url = None
    result_image_url = None
    rectangle_mask_url = None
    final_cropped_url = None
    ellipse_inference_url = None
    ellipse_fitted_url = None
    ellipse_final_url = None
    filename = None

    if request.method == 'POST':
        # ------------------------------------------------------------------
        # Delete everything from ./data at the start of a new upload
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

        # 1) Check and save the uploaded file
        if 'file' not in request.files:
            return "No file part in request."
        file = request.files['file']
        if file.filename == '':
            return "No selected file."
        filename = file.filename

        # Only BMP allowed
        if not filename.lower().endswith('.bmp'):
            return "Error: Only BMP files are accepted."

        # 2) Figure out if the user selected SW or HW inference from the radio
        #    If not provided, default to software mode
        infer_mode = request.form.get("infer_mode", "sw")  # 'sw' or 'hw'
        is_hw = (infer_mode == "hw")

        file_path = os.path.join(UPLOAD_FOLDER, INPUT_BMP)
        file.save(file_path)

        # --------------------------
        # Crop original -> data_cropped.png
        # --------------------------
        cropped_filepath = os.path.join(UPLOAD_FOLDER, CROPPED_PNG)

        with Image.open(file_path) as img:
            img_cropped = crop_image(img)
            img_cropped.save(cropped_filepath)

        # Display the cropped image
        uploaded_file_url = url_for('uploaded_file', filename=CROPPED_PNG)

        # --------------------------
        # 3) Run STEM inference (SW or HW)
        # --------------------------
        arr_out = infer_stem(cropped_filepath, hw=is_hw)
        if arr_out is None:
            # if infer_stem returns None in HW mode or an error occurred,
            # we can bail or just skip subsequent logic
            return "infer_stem returned no result."

        arr_out = arr_out.reshape(IMG_SIZE)

        # Clip [0..1], then scale up to [0..255] for display
        arr_out_clamped = np.clip(arr_out, 0.0, 1.0) * 255.0
        arr_out_clamped = arr_out_clamped.astype(np.uint8)

        # Convert to Pillow image
        result_png_filepath = os.path.join(UPLOAD_FOLDER, RESULT_PNG)
        result_img = Image.fromarray(arr_out_clamped, mode='L')
        result_img.save(result_png_filepath)
        result_image_url = url_for('uploaded_file', filename=RESULT_PNG)

        # --------------------------
        # 4) find_best_rectangle() => data_rectangle_mask.bmp
        # --------------------------
        rect_w, rect_h = 32, 32
        (best_x, best_y), best_sum = find_best_rectangle(arr_out_clamped, rect_w, rect_h)
        rect_mask = create_binary_rectangle_mask(IMG_SIZE[1], IMG_SIZE[0],
                                                 best_x, best_y, rect_w, rect_h)

        rect_mask_bmp_path = os.path.join(UPLOAD_FOLDER, RECT_MASK_BMP)
        Image.fromarray(rect_mask).save(rect_mask_bmp_path)
        rectangle_mask_url = url_for('uploaded_file', filename=RECT_MASK_BMP)

        # --------------------------
        # 5) Scale rectangle mask => bounding box => crop data_cropped.png
        # --------------------------
        with Image.open(cropped_filepath) as cropped_img:
            cw, ch = cropped_img.size

        mask_128x128_pil = Image.fromarray(rect_mask, mode='L')
        mask_scaled_pil = mask_128x128_pil.resize((cw, ch), Image.NEAREST)
        mask_scaled = np.array(mask_scaled_pil)

        bbox = get_mask_bounding_box(mask_scaled)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            with Image.open(cropped_filepath) as cropped_img:
                final_crop = crop_image_by_box(cropped_img, (xmin, ymin, xmax, ymax))
                final_cropped_path = os.path.join(UPLOAD_FOLDER, FINAL_CROPPED_PNG)
                final_crop.save(final_cropped_path)
                final_cropped_url = url_for('uploaded_file', filename=FINAL_CROPPED_PNG)

                # 5a) Now run QKeras inference on final_cropped_path
                final_cropped_const = os.path.join(UPLOAD_FOLDER, "data_cropped_final.png")
                if final_cropped_path != final_cropped_const:
                    os.replace(final_cropped_path, final_cropped_const)

                infer_ellipse()

                # Show ellipse_infer.png
                ellipse_infer_png = os.path.join(UPLOAD_FOLDER, "ellipse_infer.png")
                if os.path.exists(ellipse_infer_png):
                    ellipse_inference_url = url_for('uploaded_file', filename="ellipse_infer.png")

                # Fit ellipse around inferred ellipse
                fit_ellipse()

                # Show ellipse_fitted.png
                ellipse_fitted_png = os.path.join(UPLOAD_FOLDER, "ellipse_fitted.png")
                if os.path.exists(ellipse_fitted_png):
                    ellipse_fitted_url = url_for('uploaded_file', filename="ellipse_fitted.png")

                # Paint fitted ellipse
                paint_ellipse()

                # Show final.png
                ellipse_final_png = os.path.join(UPLOAD_FOLDER, "final.png")
                if os.path.exists(ellipse_final_png):
                    ellipse_final_url = url_for('uploaded_file', filename="final.png")

    # -----------------------
    # Return an HTML page with radio options + all relevant images
    # -----------------------
    return f'''
    <html>
        <head>
            <title>STEM Inference</title>
        </head>
        <body>
            <h1>Upload a BMP File</h1>
            <form method="POST" action="/" enctype="multipart/form-data">
                <label>
                  <input type="radio" name="infer_mode" value="sw" checked />
                  STEM UNET in SW
                </label>
                <label style="margin-left: 20px;">
                  <input type="radio" name="infer_mode" value="hw" />
                  STEM UNET in HW
                </label>
                <br><br>

                <input type="file" name="file" accept=".bmp"/>
                <input type="submit" value="Upload"/>
            </form>

            {'<p>File uploaded:</p><p>' + filename + '</p>' if filename else ''}
            {f'<img src="{uploaded_file_url}" alt="Cropped Input" style="margin-right:20px;"/>' if uploaded_file_url else ''}
            {f'<img src="{result_image_url}" alt="STEM Result" style="margin-right:20px;"/>' if result_image_url else ''}
            {f'<img src="{rectangle_mask_url}" alt="Rectangle Mask" style="margin-right:20px;"/>' if rectangle_mask_url else ''}
            {f'<img src="{final_cropped_url}" alt="Final Cropped" style="margin-right:20px;"/>' if final_cropped_url else ''}
            {f'<img src="{ellipse_inference_url}" alt="Ellipse Inference" style="margin-right:20px;"/>' if ellipse_inference_url else ''}
            {f'<img src="{ellipse_fitted_url}" alt="Fitted Ellipse" style="margin-right:20px;"/>' if ellipse_fitted_url else ''}
            {f'<img src="{ellipse_final_url}" alt="Painted Ellipse"/>' if ellipse_final_url else ''}
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
