# File: vitis_ai/stem_unet/deploy.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

'''
Vitis AI Deployment script - Auto-adaptive shape
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import math

# Globální proměnná pro ukládání masek (default)
OUTPUT_DIR = "output_masks"

def preprocess_fn(image_path, fix_height, fix_width):
    '''
    Dynamický pre-processing.
    Bere rozměry (fix_height, fix_width) podle toho, co vyžaduje model.
    '''
    # 1. Načíst jako Grayscale
    image = cv2.imread(image_path, 0) 
    
    if image is None:
        print(f"Error reading {image_path}")
        return np.zeros((fix_height, fix_width, 1), dtype=np.float32)

    # 2. Resize na vstup modelu (dynamicky)
    image = cv2.resize(image, (fix_width, fix_height), interpolation=cv2.INTER_LINEAR)
    
    # 3. Normalizace
    image = image.astype('float32') / 255.0
    
    # 4. Reshape (H, W, 1)
    image = image.reshape((fix_height, fix_width, 1))
    
    return image

def postprocess_output(output_data, filename, shape_h, shape_w):
    '''
    Dynamický post-processing.
    '''
    # Reshape podle rozměrů modelu
    prediction = output_data.reshape((shape_h, shape_w))
    
    # Binarizace (Threshold)
    mask = (prediction > 0.5).astype(np.uint8) * 255
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, "mask_" + filename)
    cv2.imwrite(save_path, mask)

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runDPU(id, start, dpu, img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    
    while count < n_of_images:
        if (count + batchSize <= n_of_images):
            runSize = batchSize
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            # Zde už chyba nebude, protože img je pre-procesovaný na správnou velikost
            imageRun[j, ...] = img[(count + j) % n_of_images]

        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = np.copy(outputData[0][j])
            write_index += 1
            
        count = count + runSize

def app(image_dir, threads, model):
    
    listimage = os.listdir(image_dir)
    # Filter only images
    listimage = [f for f in listimage if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    runTotal = len(listimage)
    
    if runTotal == 0:
        print(f"No images found in {image_dir}")
        return

    global out_q
    out_q = [None] * runTotal

    print(f"Loading model: {model}")
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    
    # Vytvoření runnerů
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # --- AUTOMATICKÁ DETEKCE ROZMĚRŮ ---
    # Získáme rozměry vstupního tensoru z prvního runneru
    inputTensors = all_dpu_runners[0].get_input_tensors()
    # Tvar je obvykle (Batch, Height, Width, Channel) -> např. (1, 256, 256, 1)
    model_h = inputTensors[0].dims[1]
    model_w = inputTensors[0].dims[2]
    
    print(f"Model expects input shape: {model_h}x{model_w}")
    # -----------------------------------

    ''' preprocess images '''
    print(f'Pre-processing {runTotal} images to {model_h}x{model_w}...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir, listimage[i])
        # Posíláme zjištěné rozměry do funkce
        img.append(preprocess_fn(path, model_h, model_w))

    '''run threads '''
    print(f'Starting {threads} threads...')
    threadAll = []
    start = 0
    for i in range(threads):
        if (i == threads - 1):
            end = len(img)
        else:
            end = start + (len(img) // threads)
        in_q = img[start:end]
        
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("---------------------------------------------")
    print(f"Throughput = {fps:.2f} FPS")
    print(f"Total frames = {runTotal}")
    print(f"Total time   = {timetotal:.4f} seconds")
    print("---------------------------------------------")

    ''' Post-processing '''
    print("Saving output masks...")
    for i in range(runTotal):
        if out_q[i] is not None:
            # I post-processing potřebuje vědět rozměry pro reshape
            postprocess_output(out_q[i], listimage[i], model_h, model_w)
            
    print(f"Done. Masks saved in '{OUTPUT_DIR}'")

    return

def main():
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images.')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads.')
  ap.add_argument('-m', '--model',     type=str, required=True,    help='Path of xmodel file.')
  # Parametr --shape už není potřeba, skript si to zjistí sám, ale pro jistotu ho tu nechám jako "dummy", aby to nespadlo, kdybys ho tam ze zvyku napsal.
  ap.add_argument('-s', '--shape',     type=int, default=0,        help='(Deprecated) Input shape is now auto-detected from the model.')

  args = ap.parse_args()  

  app(args.image_dir, args.threads, args.model)

if __name__ == '__main__':
  main()
