# Requirements

### StyleGan2-ADA-PyTorch

Partially copied from the [StyleGan2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository.

- 64-bit `Python` 3.7 and `PyTorch` 1.7.1. (But Python3.6 is also working for Nano) (Higher versions possible)
- `CUDA` toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090. (But 10.2 is also working for Nano) (Our program can also run on CPU but it's extremely slow even when loading)
- `apt install libopenmpi-dev libomp-dev libopenblas-dev pybind11-dev`
- `pip install pybind11 click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`

### Other

- `torchvision` (for nms)
- `opencv >= 4.5.4` (need `cv2.FaceDetectorYN` and I find it first appears in 4.5.4's documentation)
- `flask` (for web server)

For common computers, the above requirements should be easy to install.

On Jetson Nano, we use the following versions:
- `JetPack 4.6.1` (Ubuntu 18.04)
- `CUDA` toolkit 10.2 (Pre-installed by JetPack)
- `Python 3.6.9` (Pre-installed by JetPack)
- `PyTorch 1.9.0` (Installed from [wheel file](https://drive.google.com/file/d/1wzIDZEJ9oo62_H2oL7fYTp5_-NffCXzt/view))
- `torchvision 0.10.0` (Installed from [wheel file](https://drive.google.com/file/d/1Q2NKBs2mqkk5puFmOX_pF40yp7t-eZ32/view))
- `opencv 4.8.0` (Installed by a [script](https://jetson-docs.com/libraries/opencv/l4t32.7.1/py3.6.9))

# Directory Structure

- `code`: The code folder of our project.
    - `main`: The program folder of our project.
        - `data`: Predefined styles' pictures and numpy files.
        - `model`: The models of StyleGAN and YuNet.
        - `static`: `CSS` and `JS` files for the web server.
        - `templates`: `HTML` files for the web server.
        - `backend.py`: The backend of the program.
        - `main.py`: The program entry, interacting with Flask.
    - `stylegan2-ada-pytorch`: The StyleGan2-ADA-PyTorch repository.

# How to Run

1. Install the requirements.
2. cd to the `code/main` folder.
3. Run `python main.py <ip> <port>`, where `<ip>` is the IP address of the server and `<port>` is the port number.

We use relative paths to access files including some parts of the StyleGan2-ADA-PyTorch repository. So, please only run the program in the `main` folder.

# How to Train

1. Mostly follow the [StyleGan2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository to prepare the datasets.
2. We've modified the training script, so you can train the model directly with the following command:
    ```bash
    python train.py --outdir=<your_output_folder> --data=<your_dataset_zip_file> --gpus=1 --cfg=paper256 --mirror=1 --resume=ffhq256 --snap=10
    ```
    If you are stuck on `Evaluating metrics...`, you can refer to [this issue](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/120).