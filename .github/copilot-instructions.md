# Copilot instructions for FZ5-UNET

Purpose: give AI coding agents the precise, actionable knowledge needed to be productive quickly in this repo.

## Big picture (high-level architecture)
- Repo combines research and deployment code for a U-Net segmentation pipeline targeting both CPU/GPU and FPGA/embedded devices (FZ5/PYNQ/Vitis/HLS4ML/FINN).
- Major areas:
  - `fz5_full_model/` — primary PyTorch/Brevitas training, inference, packaging and FZ5 deployment scripts (most likely place to start).
  - `hls4ml/`, `split_model/` — FPGA-focused flows (HLS4ML, FINN, model splitting and packaging for PYNQ/Vivado).
  - `vitis_ai/` — Vitis AI graphs and build scripts for DPU targets.
  - `final_deployment/` and `util/` — glue scripts for packaging and final app deployment (includes small webapp and helper utilities).

## Quick start commands (most common tasks)
- Setup: Python 3.7+ recommended; use a virtualenv and install requirements in each area (e.g. `fz5_full_model/requirements.txt` and `requirements_inference.txt`).
- Train (CPU/GPU/Brevitas quantization-aware training):
  - cd `fz5_full_model` && `python train.py`
  - Training uses combined BCEWithLogitsLoss + Dice loss (see `train.py`).
- Inference (single image): `python infer.py <image_path>` (in `fz5_full_model`).
- Batch testbench/perf: `python testbench.py` (in `fz5_full_model`).
- Deploy to FZ5: run `fz5_full_model/deploy_fz5.sh` — it prepares `deployment/`, zips files and attempts remote tests (watch for hardcoded IPs and usernames).
- Packaging for PYNQ/HLS: see `hls4ml/*/package.sh` and notebooks under `util/unet-z2.ipynb` and `split_model/unet-z2.ipynb` for the FINN/HLS flows.

## Project-specific conventions & patterns (important to preserve)
- File names matter: `best_unet_weights.pth`, `X_test.npy`, `X_test.bin`, and `01_mask.png` are used by packaging and test scripts — avoid renaming unless you update callers.
- Data prep uses color-based mask extraction (green masks by default). See `fz5_full_model/dataprep.py` — constants like `CROP_MARGIN = 110` and `TOLERANCE = 70` are important parameters.
- Quantization path uses Brevitas (`model_quant.py`, `model.py`) and sometimes exports for FINN/HLS; follow notebooks to replicate transforms.
- Tests and perf rely on deterministic exported inputs: many scripts export `X_test.npy` and/or raw `X_test.bin` for DMA transfers — follow the expected shapes and dtypes when modifying preprocessing.

## Integration points & external dependencies
- Python libs: PyTorch, Brevitas, OpenCV, NumPy, Matplotlib (see `requirements*.txt` in each folder).
- FPGA toolchains & runtimes: HLS4ML, FINN, Vitis AI, Vivado/PYNQ — these flows are in `hls4ml/`, `split_model/`, and `vitis_ai/`.
- Remote deployment uses `scp` / SSH to specific IPs (e.g., 85.70.252.121, 192.168.50.204) and petalinux/PYNQ accounts; verify credentials & target availability.
- DMA tests use `dma_driver.c`/`dma_benchmark.c` (under `hls4ml/stem_runet/`) and `send_data.sh` to copy `X_test.bin` to the board's memory.

## Debugging & validation tips
- For model/local checks: use the included quick tests in `model.py`/`model_quant.py` (`test_input = torch.randn(...)` blocks); they’re a fast smoke-test.
- Use `testbench.py` to validate end-to-end inference and get timing stats before deploying to hardware.
- When debugging FPGA flows: reproduce the exact input export used by the driver (`X_test.bin`) and run `dma_driver`/`dma_benchmark` to validate memory transfers.
- Watch for hardcoded paths and IP addresses in `deploy_fz5.sh`, `copy_to_fz5.sh`, `send_data.sh` — update only after confirming remote OS layout.

## Where to look for examples & reference code
- Training & inference: `fz5_full_model/train.py`, `fz5_full_model/infer.py`, `fz5_full_model/testbench.py`.
- Data prep: `fz5_full_model/dataprep.py` (masks, crop parameters).
- Packaging & deployment: `fz5_full_model/deploy_fz5.sh`, `util/copy_to_fz5.sh`, `final_deployment/deploy.sh`, `hls4ml/*/package.sh`.
- FPGA DMA tests: `hls4ml/stem_runet/dma_driver.c`, `dma_benchmark.c`, `send_data.sh`.
- FINN/HLS transforms and partitioning: `split_model/unet-z2.ipynb` and `util/unet-z2.ipynb`.

## Guidance for code changes (rules for AI agents)
- Preserve backward-compatible input/output filenames and shapes unless you update all callers, tests, and packaging scripts.
- Prefer adding new helper scripts over editing working deploy/test scripts unless you can test on hardware or have a reproducible emulation path.
- Add or update unit-style smoke tests where present (many modules include small `if __name__ == '__main__'` checks) and document expected I/O in the docstring.
- When touching hardware flows, include a short checklist in your PR: (1) local unit tests pass, (2) `testbench.py` results are comparable, (3) deployment packaging still produces `X_test.npy`/`X_test.bin` and test scripts run on-board.

## Questions & follow-up
- If anything here is unclear or you want specific sections expanded (e.g., FINN/HLS flow steps, or list of hardcoded IPs), tell me which area and I’ll improve the file with examples or step-by-step commands.

---
*Generated by an automated repo scan — please review for accuracy and local details before running hardware deployment commands.*
