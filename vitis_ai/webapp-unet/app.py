import os

# File: --- KONFIGURACE ---
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect
import xir
import vart
import threading

# --- KONFIGURACE ---
MODEL_PATH = "model_dir/unet.xmodel" # Zkontroluj cestu
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- GLOBﾃ´Nﾃ・PROMﾄ哢Nﾃ・---
# Musﾃｭme drﾅｾet reference, aby je Python Garbage Collector nesmazal
g_graph = None       # <--- DﾅｮLEﾅｽITﾃ・ Graf musﾃｭ bﾃｽt globﾃ｡lnﾃｭ
dpu_runner = None
input_ndim = None
output_ndim = None
model_h = 0
model_w = 0
dpu_lock = threading.Lock() 

def init_dpu():
    # Pouﾅｾijeme globﾃ｡lnﾃｭ promﾄ嬾nﾃｩ
    global dpu_runner, input_ndim, output_ndim, model_h, model_w, g_graph
    
    print(f"[INFO] Loading model: {MODEL_PATH}")
    # Naﾄ衡eme graf do globﾃ｡lnﾃｭ promﾄ嬾nﾃｩ, aby pﾅ册ﾅｾil konec tﾃｩto funkce
    g_graph = xir.Graph.deserialize(MODEL_PATH)
    
    root_subgraph = g_graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    
    # Najﾃｭt DPU subgraph
    dpu_subgraph = [cs for cs in child_subgraphs 
                    if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"][0]

    # Vytvoﾅ冓t runner
    dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    # Zjistit rozmﾄ孑y
    inputTensors = dpu_runner.get_input_tensors()
    outputTensors = dpu_runner.get_output_tensors()
    
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    
    model_h = inputTensors[0].dims[1]
    model_w = inputTensors[0].dims[2]
    
    print(f"[INFO] Model loaded. Input shape: {model_h}x{model_w}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0) # Grayscale
    if img is None: return None
    
    img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype('float32') / 255.0
    img = img.reshape((1, model_h, model_w, 1))
    return img

def run_inference(processed_img):
    # Dﾅｯleﾅｾitﾃｩ: Vytvﾃ｡ﾅ凖ｭme C-contiguous pole pﾅ凖ｭmo zde
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]
    
    # Kopﾃｭrovﾃ｡nﾃｭ dat (zajistﾃｭ sprﾃ｡vnﾃｩ rozloﾅｾenﾃｭ pamﾄ孚i)
    inputData[0][0, ...] = processed_img[0]
    
    with dpu_lock:
        # Volﾃ｡nﾃｭ DPU
        job_id = dpu_runner.execute_async(inputData, outputData)
        dpu_runner.wait(job_id)
    
    return outputData[0][0]

def postprocess_result(output_data, filename):
    prediction = output_data.reshape((model_h, model_w))
    mask = (prediction > 0.5).astype(np.uint8) * 255
    
    result_filename = "mask_" + filename
    save_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(save_path, mask)
    
    return result_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Proces
            input_data = preprocess_image(filepath)
            if input_data is not None:
                raw_output = run_inference(input_data)
                mask_filename = postprocess_result(raw_output, filename)
                
                return render_template('index.html', 
                                       original_image=filepath, 
                                       mask_image=os.path.join(app.config['RESULT_FOLDER'], mask_filename))
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    init_dpu()
    
    # DﾅｮLEﾅｽITﾃ・PRO FLASK + DPU:
    # 1. debug=False (debug mode vytvﾃ｡ﾅ凖ｭ child procesy, kterﾃｩ rozbﾃｭjﾃｭ XRT kontext)
    # 2. threaded=True je default, ale s DPU zﾃ｡mkem (dpu_lock) je to bezpeﾄ肱ﾃｩ.
    app.run(host='0.0.0.0', port=5000, debug=False)

