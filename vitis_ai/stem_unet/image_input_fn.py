# File: vitis_ai/stem_unet/image_input_fn.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import os
import cv2

def calib_input(iter):
    HEIGHT=256
    WIDTH=256
    images = []
    images_dir = './data/images'
    for image_name in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        # Open image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Resize to (HEIGHT, WIDTH)
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            # Convert to float32 and scale to [0, 1]
            image = image.astype('float32') / 255.0
            # Add channel dimension to match (H, W, 1)
            image = image[..., None]
            images.append(image)
    return {"cnn_input": images}

