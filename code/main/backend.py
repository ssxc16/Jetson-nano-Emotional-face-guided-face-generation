import os
import sys

sys.path.append(f'{os.path.abspath("..")}/stylegan2-ada-pytorch') # to use legacy and dnnlib package

from time import time
from threading import Condition, Thread, Lock
import cv2
import numpy as np
import torch
from torchvision.ops import nms
import legacy
import dnnlib
# import jetson_utils

video_capture_args = ['nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER]
# video_capture_args = []

root_dir = os.path.abspath('.')
data_dir = f'{root_dir}/data'
stylegan_model = f'{root_dir}/model/stylegan/ffhq-res256-mirror-paper256-noaug.pkl'
yunet_model = f'{root_dir}/model/yunet/face_detection_yunet_2023mar_int8.onnx'
style_in_path = [f'{data_dir}/style{i}.png' for i in range(1,4)]
pretrained_style_npz = [f'{data_dir}/style{i}.npz' for i in range(1,4)]
hair_npz = f'{data_dir}/white_hair.npz'
beard_npz = f'{data_dir}/beard.npz'
smile_npz = f'{data_dir}/smile.npz'

class StyleGAN:
    def __init__(self, network_pkl):
        self.network_pkl = network_pkl
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.layer_style = [range(7,14)]
        self.layer_hair = [range(6,13)]
        self.layer_beard = [range(4,14)]
        self.layer_smile = [range(3,5)]

        self.w_hair = torch.tensor(np.load(hair_npz)['w'], device=self.device)
        self.w_beard = torch.tensor(np.load(beard_npz)['w'], device=self.device)
        self.w_smile = torch.tensor(np.load(smile_npz)['w'], device=self.device)
        self.G = None

    def init_model(self):
        print('Loading networks from "%s"...' % self.network_pkl)
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(self.device)

    def generate_by_w(self, w: torch.tensor, noise_mode: str = 'const') -> np.ndarray:
        # w = self.G.mapping(
        #     torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device), None
        # )
        img = self.G.synthesis(w, noise_mode=noise_mode)[:,[2,1,0],:,:]
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    def mix_by_latent(self, latent, style, layer_style=None):
        if layer_style is None:
            layer_style = self.layer_style
        w = latent.clone()
        w[0][layer_style] = style[0][layer_style]
        return w

    # larger alpha, more styled
    def mix_hair_gradually(self, latent, alpha):
        latent[0][self.layer_hair] = latent[0][self.layer_hair] * (1-alpha) + self.w_hair[0][self.layer_hair] * alpha

    def mix_beard_gradually(self, latent, alpha):
        latent[0][self.layer_beard] = latent[0][self.layer_beard] * (1-alpha) + self.w_beard[0][self.layer_beard] * alpha
    
    def mix_smile_gradually(self, latent, alpha):
        latent[0][self.layer_smile] = latent[0][self.layer_smile] * (1-alpha) + self.w_smile[0][self.layer_smile] * alpha

class Backend:
    BLACK_JPEG = cv2.imencode('.jpg', np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()

    def __init__(self):
        self.camera_buffer = [None] * 2
        self.camera_buffer_idx = 0
        self.camera_buffer_cv = Condition()

        self.raw_buffer = None
        self.map_buffer = None
        self.gan_buffer = None
        self.captured_buffer = None
        self.style_in_buffer = None
        self.style_out_buffer = None

        self.mapping_buffer = None
        # self.mapping_buffer_cuda = None

        self.video_buffer_cv = Condition()
        self.video_gan_cv = Condition()

        self.need_capture = False
        self.need_capture_fail_times = 0
        self.need_capture_cv = Condition()

        self.stylegan = StyleGAN(stylegan_model)
        self.stylegan_cv = Condition()
        self.stylegan_cv_req = False # urgent request

        self.map_w_raw = None # 左边模式的w
        self.map_w_mix = None
        self.map_w_mix_lock = Lock()
        self.project_w = None # 右边模式的w
        self.predefined_styles = [torch.tensor( np.load(path)['w']).to(self.stylegan.device) for path in pretrained_style_npz]
        self.map_styles = {'hair':0.0, 'beard':0.0, 'smile':0.0}

        self.running = False
        self.threads = [
            Thread(target=self._camera_thread),
            Thread(target=self._capture_thread),
            Thread(target=self._map_thread),
        ]

        self.cap = None

        self.yunet = cv2.FaceDetectorYN.create(
            model=yunet_model,
            config='',
            input_size=(320, 320),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        self.yunet.setInputSize((320, 320))

        self.stylegan.init_model()

        self.style_in_buffer = [None] * len(style_in_path)
        self.style_out_buffer = [None] * len(self.predefined_styles)
        for i in range(len(style_in_path)):
            img = cv2.imread(style_in_path[i])
            self.style_in_buffer[i] = cv2.imencode('.jpg', img)[1].tobytes()
        self.update_seed(-567)
        self.recv_w(self.map_w_raw)
        self.captured_buffer = cv2.imencode('.jpg', self.stylegan.generate_by_w(self.map_w_raw))[1].tobytes()

    def start(self):
        self.cap = cv2.VideoCapture(*video_capture_args)
        if not self.cap.isOpened() or not self.cap.read()[0]:
            print("Camera open failed")
            exit(0)
        self.running = True
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()
        self.cap.release()

    def is_video_ready(self):
        return self.map_buffer is not None

    def is_video_gan_ready(self):
        return self.gan_buffer is not None

    def get_raw_buffer(self):
        return self.raw_buffer

    def get_map_buffer(self):
        return self.map_buffer

    def get_gan_buffer(self):
        return self.gan_buffer

    def get_captured_buffer(self):
        return self.captured_buffer if self.captured_buffer is not None else Backend.BLACK_JPEG

    def get_style_in_buffer(self, idx):
        return self.style_in_buffer[idx] if idx < len(self.style_in_buffer) and self.style_in_buffer[idx] else Backend.BLACK_JPEG

    def get_style_out_buffer(self, idx):
        return self.style_out_buffer[idx] if idx < len(self.style_out_buffer) and self.style_out_buffer[idx] else Backend.BLACK_JPEG

    def require_capture(self):
        # FIXME: just for test
        # self.stylegan_cv_req = True
        # with self.stylegan_cv:
        #     random_noise = torch.randn(1, self.stylegan.G.z_dim, device=self.stylegan.device)
        #     captured = self.stylegan.generate_by_w(self.stylegan.G.mapping(random_noise, None))
        #     self.captured_buffer = cv2.imencode('.jpg', captured)[1].tobytes()
        #     self.stylegan_cv_req = False
        #     return True

        with self.need_capture_cv:
            self.need_capture = True
            self.need_capture_fail_times = 0
            self.need_capture_cv.wait()
            return self.captured_buffer is not None

    def video_next(self):
        with self.video_buffer_cv:
            self.video_buffer_cv.wait()

    def video_gan_next(self):
        with self.video_gan_cv:
            self.video_gan_cv.wait()

    def generated(self):
        return self.project_w is not None

    def update_seed(self, seed):
        torch.manual_seed(seed)
        random_noise = torch.randn(1, self.stylegan.G.z_dim, device=self.stylegan.device)
        self.map_w_raw = self.stylegan.G.mapping(random_noise, None)
        with self.map_w_mix_lock:
            self.map_w_mix = None

    def update_style(self, key, value):
        if key not in self.map_styles:
            return False
        self.map_styles[key] = np.clip(value, 0, 1)
        with self.map_w_mix_lock:
            self.map_w_mix = None
        return True

    def save_w(self):
        if self.project_w is None:
            return False
        self.map_w_raw = self.project_w
        with self.map_w_mix_lock:
            self.map_w_mix = None
        return True

    def recv_w(self, w):
        self.project_w = w
        self.stylegan_cv_req = True
        with self.stylegan_cv:
            for i in range(len(self.predefined_styles)):
                styled_latent = self.stylegan.mix_by_latent(self.project_w, self.predefined_styles[i])
                styled_img = self.stylegan.generate_by_w(styled_latent)
                self.style_out_buffer[i] = cv2.imencode('.jpg', styled_img)[1].tobytes()
            self.stylegan_cv_req = False
            self.stylegan_cv.notify_all()

    def get_fit_rect(rect):
        x, y, w, h = rect
        if w > h:
            y -= (w - h) // 2
            h = w
        else:
            x -= (h - w) // 2
            w = h
        return x, y, w, h

    def _camera_thread(self):
        while self.running:
            success, image = self.cap.read()
            if not success:
                print("Camera read failed")
                break
            with self.camera_buffer_cv:
                self.camera_buffer[self.camera_buffer_idx] = image

    def _capture_thread(self):
        # cap = jetson_utils.videoSource("csi://0", ['--input-width=640', '--input-height=360', '--input-rate=30'])
        last_time = time()
        fps = 0
        # image_resized_cuda = jetson_utils.cudaAllocMapped(width=320, height=180, format='rgb8')
        # square_image_cuda = jetson_utils.cudaAllocMapped(width=320, height=320, format='rgb8')
        square_image = np.zeros((320, 320, 3), dtype=np.uint8)
        while self.running:
            # image_cuda = cap.Capture()
            # try:
            #     image = jetson_utils.cudaToNumpy(image_cuda)
            # except:
            #     continue
            # jetson_utils.cudaDeviceSynchronize()
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # jetson_utils.cudaResize(image_cuda, image_resized_cuda)
            # jetson_utils.cudaOverlay(image_resized_cuda, square_image_cuda, 0, 70)
            # _, detections = net.detect(jetson_utils.cudaToNumpy(square_image_cuda))
            # detections = None
            with self.camera_buffer_cv:
                image = self.camera_buffer[self.camera_buffer_idx]
                self.camera_buffer[self.camera_buffer_idx] = None
                if image is None:
                    continue
                self.camera_buffer_idx = 1 - self.camera_buffer_idx
            square_image[70:250, :] = cv2.resize(image, (320, 180))
            _, detections = self.yunet.detect(square_image)
            if detections is None:
                detections = np.empty(shape=(0, 5))
            bboxs = detections[:, 0:4]
            confs = detections[:, -1]
            if len(bboxs) > 1:
                bboxs = bboxs[nms(torch.from_numpy(bboxs), torch.from_numpy(confs), 0.5), :]
            else:
                bboxs = bboxs
            bboxs = bboxs.astype(np.int32)
            bboxs[:, 1] -= 70
            bboxs *= 2
            np.clip(bboxs[:, 0], 0, 640, out=bboxs[:, 0])
            np.clip(bboxs[:, 1], 0, 360, out=bboxs[:, 1])
            np.clip(bboxs[:, 2], 0, 640 - bboxs[:, 0], out=bboxs[:, 2])
            np.clip(bboxs[:, 3], 0, 360 - bboxs[:, 1], out=bboxs[:, 3])
            highest_conf_box = bboxs[np.argmax(confs)] if len(confs) > 0 else None
            with self.need_capture_cv:
                if self.need_capture:
                    if highest_conf_box is not None:
                        x, y, w, h = Backend.get_fit_rect(highest_conf_box)
                        x, y = max(0, x), max(0, y)
                        captured = image[y:y+h, x:x+w]
                        if 0 not in captured.shape:
                            self.captured_buffer = cv2.imencode('.jpg', captured)[1].tobytes()
                            self.need_capture = False
                            self.need_capture_cv.notify_all()
                    elif self.need_capture_fail_times >= 5:
                        self.captured_buffer = None
                        self.need_capture = False
                        self.need_capture_cv.notify_all()
                    self.need_capture_fail_times += 1
            for bbox, conf in zip(bboxs, confs):
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                # cv2.putText(image, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.raw_buffer = cv2.imencode('.jpg', image)[1].tobytes()

            # if self.mapping_buffer_cuda is not None:
            if self.mapping_buffer is not None:
                mapping_buffer = self.mapping_buffer
                for bbox, conf in zip(bboxs, confs):
                    # resized_map = jetson_utils.cudaAllocMapped(width=bbox[2], height=bbox[3], format='bgr8')
                    # jetson_utils.cudaResize(self.mapping_buffer_cuda, resized_map)
                    # jetson_utils.cudaOverlay(resized_map, image_cuda, bbox[0], bbox[1])
                    image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = cv2.resize(mapping_buffer, (bbox[2], bbox[3]))
            self.map_buffer = cv2.imencode('.jpg', image)[1].tobytes()

            with self.video_buffer_cv:
                self.video_buffer_cv.notify_all()

            now_time = time()
            fps = 1 / (now_time - last_time)
            last_time = now_time

    def _map_thread(self):
        while self.running:
            with self.map_w_mix_lock:
                map_w_mix = self.map_w_mix
                if map_w_mix is None:
                    map_w_mix = self.map_w_raw.clone()
                    self.stylegan.mix_hair_gradually(map_w_mix, self.map_styles['hair'])
                    self.stylegan.mix_beard_gradually(map_w_mix, self.map_styles['beard'])
                    self.stylegan.mix_smile_gradually(map_w_mix, self.map_styles['smile'])
                    self.map_w_mix = map_w_mix
            with self.stylegan_cv:
                if self.stylegan_cv_req:
                    self.stylegan_cv.wait()
                self.mapping_buffer = self.stylegan.generate_by_w(map_w_mix, noise_mode='random')
                # self.mapping_buffer_cuda = jetson_utils.cudaImage(ptr=tensor.data_ptr(), width=tensor.shape[-2], height=tensor.shape[-3], format='bgr8')
            self.gan_buffer = cv2.imencode('.jpg', self.mapping_buffer)[1].tobytes()
            with self.video_gan_cv:
                self.video_gan_cv.notify_all()
