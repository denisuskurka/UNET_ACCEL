# UNET FPGA Acceleration

![EU sponsorship](evropa_logo.png) ![NPO sponsorship](npo_logo.png)

**Author:** Denis Kurka

A concise repository for accelerating a U-Net segmentation model on FPGA platforms. This project contains two main, fully-working flows and several experimental/archived efforts for reference and comparison.

## Project layout (short)
- `hls4ml/` — Accelerate UNET using HLS4ML for PYNQ/Vivado targets; includes example packaging and a working webapp in `final_deployment/` used for PYNQ deployments.
- `vitis_ai/` — Accelerate UNET using Vitis AI (DPU). Primary target: Kria KV260. Includes `webapp-recycle` (the Vitis AI webapp used for demos/comparison).
- `fz5_full_model/` — Original FZ5 deployment and Brevitas quantization-aware training code; includes `deploy_fz5.sh` and `testbench.py` for packaging and remote testing.
- `split_model/`, `archive/` — Experimental or archived projects (split UNET experiments and an older PaddlePaddle branch). Keep for reference.
- `final_deployment/` — Webapp packaging for the HLS4ML flow and helper scripts for deployment.

---

## Quick prerequisites
- Python 3.7+ (virtualenv recommended)
- PyTorch, Brevitas, OpenCV, NumPy, Matplotlib (see `fz5_full_model/requirements.txt` and subfolder requirements)
- Vivado / Vitis / Vitis-AI toolchains for FPGA/DPU builds (where applicable)
- HLS4ML and FINN (for FPGA streaming flows)

Notes: Hardware deployment requires SSH/SCP access to the target board (some scripts contain **hardcoded IPs** — check `deploy_fz5.sh`, `copy_to_fz5.sh`, and `send_data.sh` before use).

---

## How to train and deploy to the FZ5 (final_deployment flow)
This flow uses the code in `fz5_full_model/` and `final_deployment/` to train, package, and run a webapp on an FZ5-class board.

1. Prepare environment
   - Create a virtualenv and install requirements for `fz5_full_model`:
     ```bash
     cd fz5_full_model
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

2. Train the model
   - Train a quantized UNet (Brevitas-aware training):
     ```bash
     python train.py
     ```
   - Trained weights are saved (watch for `best_unet_weights.pth`). Keep this filename unless you update callers.

3. Validate locally
   - Run a quick smoke test of the model and exports:
     ```bash
     python testbench.py
     python infer.py path/to/image.png
     ```

4. Create deployment package for FZ5
   - The `deploy_fz5.sh` script prepares `deployment/`, copies example inputs (e.g., `test_data/01.png`) and creates a zip for scp/transfer.
     ```bash
     bash deploy_fz5.sh
     ```
   - IMPORTANT: Open `deploy_fz5.sh` and verify the remote SSH settings (IP, username, port). Example IPs appear in scripts (e.g., 85.70.252.121, 192.168.50.204).

5. Remote board-side test
   - Unzip the package on the board, run the included inference script and verify output (many scripts create `01_mask.png` or `Y_test.bin` depending on flow).
   - For HLS/DMA validation (if applicable) use `hls4ml/*/send_data.sh` and the `dma_driver`/`dma_benchmark` binaries to copy `X_test.bin` and retrieve `Y_test.bin`.

6. Webapp
   - The HLS4ML-based webapp used in demos is in `final_deployment/`. Deploy it to the board following `final_deployment/deploy.sh` (edit SSH details first).

---

## How to train and run the Vitis-AI `webapp-recycle`
This flow targets DPU/KV260 using Vitis-AI tooling; it uses the same UNet architecture so results can be compared with the HLS4ML/FZ5 flows.
If you don't want to train your own NN, you can just skip to step 4. for webapp-recycle deployment on the Kria KV260.

1. Environment
   - Install/prepare Vitis-AI toolchain and board support packages for your target (KV260). Follow Xilinx Vitis-AI docs for setup and device installation.

2. Quantize & evaluate
   - Each Vitis model folder contains scripts like `5_eval_quant.sh` for evaluation and quantization.
     ```bash
     cd vitis_ai/<model_folder>
     ./5_eval_quant.sh
     ```
   - Run each script starting from ./0_.. to  the last one.
   - Just can change the network/training properties in the 0 script.

3. Copy final models to the webapp-recycle root folder (there should already be models like .xmodel and .h5).
   - Warning! only same resolution networks can be used in one pipeline! So if you trained 256x256 stem unet, you need 256x256 ellipse regressor as well.

4. Deploy and run the `webapp-recycle`
   - After all models are ready in the folder, zip the whole webapp-recylce folder.
   - Copy to the target KV260 board and unzip.
   - Install all requirements in the requirements.txt or use the provided image.
   - run python3 app.py

5. Notes on comparison
   - Because the same model architecture is used across flows, you can directly compare inference accuracy and latency between the HLS4ML (PYNQ/FZ5) and Vitis AI (KV260) flows. Keep identical test inputs (`X_test.npy`/`X_test.bin`) and post-processing.

---

## Archived / experimental projects
- `archive/paddlepaddle-old/` — older experiments and a Paddle-Paddle conversion flow (kept for historical reference).
- `split_model/` — experiments with splitting the UNET (model partitioning) to improve FPGA performance; not production-ready but useful for reference and testing.
- Keep archived folders read-only unless you are intentionally trying to reproduce or extend those experiments.

---

## Conventions & important notes
- Preserve key filenames: `best_unet_weights.pth`, `X_test.npy`, `X_test.bin`, `01_mask.png` unless you update all callers.
- Mask extraction is color-based (green default) — see `fz5_full_model/dataprep.py` for constants: `CROP_MARGIN = 110`, `TOLERANCE = 70`.
- Hardware flows rely on exported inputs (`X_test.npy` / `X_test.bin`) and sometimes expect specific dtypes and shapes — when changing preprocessing, update test exports and DMA drivers accordingly.
- Many deploy scripts contain hardcoded IPs and usernames; **verify and edit before running remote operations**.

---

## Contact / Author
- Denis Kurka
- Petr Čermák

---

## License
See the `LICENSE` file in this repository root for license details.
