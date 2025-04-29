
<div align="center">
<h1>Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation beyond 3D Training Data</h1>

<div>
    <a href='https://scholar.google.com/citations?user=F15mLDYAAAAJ&hl=en' target='_blank'>Zhiyuan Ma</a>&emsp;
    <a href='https://scholar.google.com/citations?user=R9PlnKgAAAAJ&hl=en' target='_blank'>Xinyue Liang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=en' target='_blank'>Rongyuan Wu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=1rbNk5oAAAAJ&hl=zh-CN' target='_blank'>Xiangyu Zhu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=cuJ3QG8AAAAJ&hl=en' target='_blank'>Zhen Lei</a>&emsp;
    <a href='https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en' target='_blank'>Lei Zhang</a>
</div>


[[Paper]](https://arxiv.org/pdf/2406.08177)
[[HF Demo]](https://huggingface.co/spaces/ZhiyuanthePony/TriplaneTurbo)

---

</div>


## 🔥 News

- **2025-03-13**: The source code and pretrained models are released.
- **2025-03-03**: Gradio and HuggingFace Demos are available.
- **2025-02-27**: TriplaneTurbo is accepted to CVPR 2025.

## 🌟 Start local inference in 3 minutes

```python
python -m venv venv
source venv/bin/activate
bash setup.sh
python gradio_app.py
```

## ⚙️ Dependencies and Installation for Training
<details>
<summary> Click to expand instructions </summary>

1.  **Clone the necessary repositories:**
    Ensure you have cloned this repository (`TriplaneTurbo_v1`) and the `3dgrut` repository into your project's root directory.
    ```sh
    # If you haven't already cloned 3dgrut:
    git clone https://github.com/nv-tlabs/3dgrut.git
    ```

2.  **Create and activate the Conda environment:**
    We recommend creating a new environment named `triplaneturbo` with Python 3.11, as required by the `3dgrut` dependency.
    ```sh
    conda create -n triplaneturbo python=3.11 -y
    conda activate triplaneturbo
    ```

3.  **Install dependencies:**
    Run the following commands sequentially to install PyTorch, specific versions of extensions, and other requirements:
    ```sh
    # Install PyTorch with CUDA 12.1
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Install compatible xformers and ninja using pip
    pip install xformers==0.0.25 
    pip install ninja
    
    # Clone and install diff-gaussian-rasterization submodule, then remove the cloned directory
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization --recursive && pip install ./diff-gaussian-rasterization && rm -rf diff-gaussian-rasterization

    # Install requirements from 3dgrut repository
    pip install -r ./3dgrut/requirements.txt
    
    # Install tiny-cuda-nn and nerfacc
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
    
    # Install custom CUDA extensions (KNN and Frequency Encoding)
    # Ensure you are in the project root directory (TriplaneTurbo_v1)
    (cd custom/primiturbo/extern/knn && python setup.py install)
    (cd custom/primiturbo/extern/kdn && python setup.py install)
    (cd custom/primiturbo/extern/frequency_encoding && python setup.py install)
    
    # Install the 3dgrut library itself
    pip install ./3dgrut
    
    # Install main project requirements
    pip install -r requirements.txt


    ```

    *Note on GCC version for tiny-cuda-nn:* If you encounter issues installing `tiny-cuda-nn`, you might need a specific GCC version (e.g., 9.5.0). You can install it within your conda environment using `conda install -c conda-forge gxx=9.5.0` before running the `pip install tiny-cuda-nn` command.

</details>

<details>
<summary> Download Pretrained Models. </summary>

```python
python scripts/prepare/download.py
```
</details>

<details>
<summary> Download Training Corpus. </summary>

```python
python scripts/prepare/download.py
```
</details>

