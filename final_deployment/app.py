import os
import numpy as np
from flask import Flask, request, url_for, send_from_directory
from PIL import Image

from util import crop_image
from rects_from_masks import find_best_rectangle, create_binary_rectangle_mask
from crop_rect_from_original import scale_mask_to_original, get_mask_bounding_box, crop_image_by_box
# The below script references final_cropped image as INPUT_PIC and writes ellipse_infer.bmp as OUTPUT_EXPORT
from infer_ellipse import infer_ellipse  
from fit_ellipse import fit_ellipse
from paint_ellipse import paint_ellipse

app = Flask(__name__)

UPLOAD_FOLDER = './data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (128, 128)
INPUT_BMP = "data.bmp"
CROPPED_BMP = "data_cropped.bmp"
STEM_INPUT_BMP = "data_stem_input.bmp"
STEM_INPUT_BIN = "data_stem_input.bin"
RESULT_BIN = "result.bin"
RESULT_BMP = "data_result.bmp"
RECT_MASK_BMP = "data_rectangle_mask.bmp"
FINAL_CROPPED_PNG = "data_cropped_final.png"

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
    filename = None

    if request.method == 'POST':
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

        file_path = os.path.join(UPLOAD_FOLDER, INPUT_BMP)
        file.save(file_path)

        # --------------------------
        # 2) Crop original, resize, grayscale, and save as data_stem_input.bmp
        # --------------------------
        cropped_filepath = os.path.join(UPLOAD_FOLDER, CROPPED_BMP)
        stem_input_filepath = os.path.join(UPLOAD_FOLDER, STEM_INPUT_BMP)
        binary_filepath = os.path.join(UPLOAD_FOLDER, STEM_INPUT_BIN)

        with Image.open(file_path) as img:
            # Custom crop
            img_cropped = crop_image(img)
            img_cropped.save(cropped_filepath)

            # Resize to 128×128
            img_resized = img_cropped.resize(IMG_SIZE)

            # Convert to grayscale
            img_gray = img_resized.convert('L')
            img_gray.save(stem_input_filepath)  # For preview

            # Save float32 binary for some accelerator
            arr_in = np.array(img_gray, dtype=np.float32)  # [0..255] as float32
            with open(binary_filepath, 'wb') as f:
                f.write(arr_in.tobytes())

        # Show the 128×128 grayscale input on the page
        uploaded_file_url = url_for('uploaded_file', filename=STEM_INPUT_BMP)

        # --------------------------
        # 3) (Optional) DMA -> produce result.bin => data_result.bmp
        # --------------------------
        result_bin_path = os.path.join(UPLOAD_FOLDER, RESULT_BIN)
        result_bmp_path = os.path.join(UPLOAD_FOLDER, RESULT_BMP)

        # Here you might have something like:
        # os.system('./run_dma.sh')

        if os.path.exists(result_bin_path):
            # Build a grayscale BMP from result.bin
            arr_out = np.fromfile(result_bin_path, dtype=np.float32)
            if arr_out.size == (IMG_SIZE[0] * IMG_SIZE[1]):
                arr_out = arr_out.reshape(IMG_SIZE)
                arr_out_clipped = np.clip(arr_out, 0, 255).astype(np.uint8)

                result_img = Image.fromarray(arr_out_clipped, mode='L')
                result_img.save(result_bmp_path)
                result_image_url = url_for('uploaded_file', filename=RESULT_BMP)

                # --------------------------
                # 4) find_best_rectangle() => data_rectangle_mask.bmp
                # --------------------------
                rect_w, rect_h = 32, 32
                (best_x, best_y), best_sum = find_best_rectangle(
                    arr_out_clipped, rect_w, rect_h
                )
                rect_mask = create_binary_rectangle_mask(
                    IMG_SIZE[1],  # 128 (height)
                    IMG_SIZE[0],  # 128 (width)
                    best_x, best_y,
                    rect_w, rect_h
                )

                rect_mask_bmp_path = os.path.join(UPLOAD_FOLDER, RECT_MASK_BMP)
                Image.fromarray(rect_mask).save(rect_mask_bmp_path)
                rectangle_mask_url = url_for('uploaded_file', filename=RECT_MASK_BMP)

                # --------------------------
                # 5) Scale rectangle mask => bounding box => crop data_cropped.bmp
                # --------------------------
                with Image.open(cropped_filepath) as cropped_img:
                    cw, ch = cropped_img.size

                # Scale 128x128 -> (cw, ch) with NEAREST
                mask_128x128_pil = Image.fromarray(rect_mask, mode='L')
                mask_scaled_pil = mask_128x128_pil.resize((cw, ch), Image.NEAREST)
                mask_scaled = np.array(mask_scaled_pil)

                bbox = get_mask_bounding_box(mask_scaled)
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    with Image.open(cropped_filepath) as cropped_img:
                        final_crop = crop_image_by_box(cropped_img, (xmin, ymin, xmax, ymax))
                        final_cropped_path = os.path.join(UPLOAD_FOLDER, FINAL_CROPPED_PNG)
                        # Save final crop
                        final_crop.save(final_cropped_path)
                        final_cropped_url = url_for('uploaded_file', filename=FINAL_CROPPED_PNG)

                        # 5a) Now run QKeras inference on final_cropped_path
                        # The 'infer_ellipse.py' script references fixed constants:
                        #   INPUT_PIC = "./data/data_cropped_final.bmp"
                        #   OUTPUT_EXPORT = "./data/ellipse_infer.bmp"
                        # So let's rename/copy final_cropped_path to "data_cropped_final.bmp"
                        # so the script sees the correct input.

                        final_cropped_const = os.path.join(UPLOAD_FOLDER, "data_cropped_final.png")
                        if final_cropped_path != final_cropped_const:
                            # rename or copy:
                            os.replace(final_cropped_path, final_cropped_const)

                        # Now call the inference script
                        # This will read data_cropped_final.bmp, produce ellipse_infer.bmp
                        infer_ellipse()

                        # Show ellipse_infer.bmp
                        ellipse_infer_bmp = os.path.join(UPLOAD_FOLDER, "ellipse_infer.png")
                        if os.path.exists(ellipse_infer_bmp):
                            ellipse_inference_url = url_for('uploaded_file', filename="ellipse_infer.png")

                        # Fit ellipse around infered ellipse
                        fit_ellipse()

                        # Show elipse_fitted.png
                        ellipse_fitted_bmp = os.path.join(UPLOAD_FOLDER, "ellipse_fitted.png")
                        if os.path.exists(ellipse_fitted_bmp):
                            ellipse_fitted_url = url_for('uploaded_file', filename="ellipse_fitted.png")

                        # Paint fitted ellipse
                        paint_ellipse()

                        # Show elipse_fitted.png
                        ellipse_final_bmp = os.path.join(UPLOAD_FOLDER, "final.png")
                        if os.path.exists(ellipse_final_bmp):
                            ellipse_final_url = url_for('uploaded_file', filename="final.png")


    # -----------------------
    # Return an HTML page that displays:
    #   1) The 128×128 grayscale input image
    #   2) The DMA result
    #   3) The rectangle mask
    #   4) The final cropped region
    #   5) The inference result from infer_ellipse.py
    # -----------------------
    return f'''
    <html>
        <body>
            <h1>Upload a BMP File</h1>
            <form method="POST" action="/" enctype="multipart/form-data">
                <input type="file" name="file" accept=".bmp"/>
                <input type="submit" value="Upload"/>
            </form>

            {'<p>File uploaded:</p><p>' + filename + '</p>' if filename else ''}
            {f'<img src="{uploaded_file_url}" alt="Processed Input" style="margin-right:20px;"/>' if uploaded_file_url else ''}
            {f'<img src="{result_image_url}" alt="DMA Result" style="margin-right:20px;"/>' if result_image_url else ''}
            {f'<img src="{rectangle_mask_url}" alt="Rectangle Mask" style="margin-right:20px;"/>' if rectangle_mask_url else ''}
            {f'<img src="{final_cropped_url}" alt="Final Cropped" style="margin-right:20px;"/>' if final_cropped_url else ''}
            {f'<img src="{ellipse_inference_url}" alt="Ellipse Inference"/>' if ellipse_inference_url else ''}
            {f'<img src="{ellipse_fitted_url}" alt="Ellipse Inference"/>' if ellipse_fitted_url else ''}
            {f'<img src="{ellipse_final_url}" alt="Ellipse Inference"/>' if ellipse_final_url else ''}
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
