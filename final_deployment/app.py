import os
from flask import Flask, request, url_for, send_from_directory
from PIL import Image
from util import crop_image

app = Flask(__name__)

# Directory to store uploads
UPLOAD_FOLDER = './data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fixed resolution for images
IMG_SIZE = (128, 128)

@app.route('/data/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_file_url = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in request."

        file = request.files['file']
        if file.filename == '':
            return "No selected file."

        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Check if file is an image by extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Open and process the image
            stem_input_filename = ""
            with Image.open(file_path) as img:
                # 1) Define new filenames
                name_stem, ext = os.path.splitext(filename)
                cropped_filename = f"{name_stem}_cropped{ext}"
                stem_input_filename = f"{name_stem}_stem_input{ext}"

                cropped_filepath = os.path.join(UPLOAD_FOLDER, cropped_filename)
                stem_input_filepath = os.path.join(UPLOAD_FOLDER, stem_input_filename)

                # 2) Crop
                img_cropped = crop_image(img)
                img_cropped.save(cropped_filepath)

                # 3) Resize (for NN accelerator)
                img_resized = img_cropped.resize(IMG_SIZE)
                img_resized.save(stem_input_filepath)


            # Generate a URL to display the processed image
            uploaded_file_url = url_for('uploaded_file', filename=stem_input_filename)

    # Return the same page with form + optional image preview
    return f'''
    <html>
        <body>
            <h1>Upload a File</h1>
            <form method="POST" action="/" enctype="multipart/form-data">
                <input type="file" name="file"/>
                <input type="submit" value="Upload"/>
            </form>
            {'<p>File uploaded:</p><p>' + filename + '</p>' if filename else ''}
            {f'<img src="{uploaded_file_url}" alt="Processed Image"/>' if uploaded_file_url else ''}
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Run the Flask app for local testing
    app.run(debug=True)
