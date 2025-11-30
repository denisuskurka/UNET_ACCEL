import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for, redirect
import xir
import vart
import threading
import time

# --- KONFIGURACE ---
MODEL_PATH = "model_dir/unet.xmodel"  # Uprav dle názvu souboru
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Globální proměnné pro DPU
dpu_runner = None
input_ndim = None
output_ndim = None
model_h = 0
model_w = 0
dpu_lock = threading.Lock() # Zámek pro thread-safety

def init_dpu():
    global dpu_runner, input_ndim, output_ndim, model_h, model_w
    print(f"[INFO] Loading model: {MODEL_PATH}")
    g = xir.Graph.deserialize(MODEL_PATH)
    
    # Najít DPU subgraph
    root_subgraph = g.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    dpu_subgraph = [cs for cs in child_subgraphs 
                    if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"][0]

    # Vytvořit runner
    dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    # Zjistit rozměry vstupu
    inputTensors = dpu_runner.get_input_tensors()
    outputTensors = dpu_runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims) # (Batch, H, W, C)
    output_ndim = tuple(outputTensors[0].dims)
    
    model_h = inputTensors[0].dims[1]
    model_w = inputTensors[0].dims[2]
    
    print(f"[INFO] Model loaded. Input shape: {model_h}x{model_w}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Načíst černobíle pro model
    img = cv2.imread(image_path, 0)
    if img is None: return None
    
    # Resize na vstup modelu
    img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalizace a reshape
    img = img.astype('float32') / 255.0
    img = img.reshape((1, model_h, model_w, 1)) # Batch size 1
    return img

def run_inference(processed_img):
    # Vytvoření bufferů
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]
    print("vlozeni dat")
    # Vložení dat
    inputData[0][0, ...] = processed_img[0]
    print("spoustim DPU")
    # Spuštění DPU (synchronně pod zámkem)
    with dpu_lock:
        print("job_id")
        job_id = dpu_runner.execute_async(inputData, outputData)
        print("dpu_runner")
        dpu_runner.wait(job_id)
    
    return outputData[0][0]

def postprocess_result(output_data, filename):
    # Reshape z (H, W, 1) na (H, W)
    prediction = output_data.reshape((model_h, model_w))
    
    # Práhování (Binarizace)
    mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Uložení
    result_filename = "mask_" + filename
    save_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(save_path, mask)
    
    return result_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kontrola zda byl nahrán soubor
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # 1. Uložit originál
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 2. Předzpracování
            input_data = preprocess_image(filepath)
            
            # 3. Inference
            raw_output = run_inference(input_data)
            
            # 4. Postprocessing a uložení masky
            mask_filename = postprocess_result(raw_output, filename)
            
            # 5. Zobrazení výsledku
            return render_template('index.html', 
                                   original_image=filepath, 
                                   mask_image=os.path.join(app.config['RESULT_FOLDER'], mask_filename))

    return render_template('index.html')

if __name__ == '__main__':
    # Vytvoření složek pokud neexistují
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    # Inicializace DPU
    init_dpu()
    
    # Spuštění serveru (host='0.0.0.0' zpřístupní web v síti)
    app.run(host='0.0.0.0', port=5000, debug=False)
