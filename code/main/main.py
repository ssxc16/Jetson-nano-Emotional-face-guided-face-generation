from threading import Condition
from time import sleep
from flask import Flask, render_template, request, Response, jsonify
import io
import sys
import numpy as np
import torch
import cv2
import requests
import base64
import traceback

from backend import Backend

if len(sys.argv) > 2:
    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
else:
    server_ip = '10.20.148.190'
    server_port = 5000
latent_server_url = '10.20.147.91:6100'
latent_server_waiting = False
latent_server_cv = Condition()
app = Flask(__name__)

print("Loading backend...")
backend = Backend()
print("Backend loaded")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(get_buffer):
    while True:
        if not backend.is_video_ready():
            sleep(1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + get_buffer() + b'\r\n')
        backend.video_next()

def gen_gan_frames():
    while True:
        if not backend.is_video_gan_ready():
            sleep(1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + backend.get_gan_buffer() + b'\r\n')
        backend.video_gan_next()

@app.route('/video_raw')
def video_raw():
    return Response(gen_frames(backend.get_raw_buffer), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_map')
def video_map():
    return Response(gen_frames(backend.get_map_buffer), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_gan')
def video_gan():
    return Response(gen_gan_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/captured_image')
def captured_image():
    return Response(backend.get_captured_buffer(), mimetype='image/jpeg')

@app.route('/style<id>_in')
def style_in(id):
    return Response(backend.get_style_in_buffer(int(id)), mimetype='image/jpeg')

@app.route('/style<id>_out')
def style_out(id):
    return Response(backend.get_style_out_buffer(int(id)), mimetype='image/jpeg')

@app.route('/bnt_state')
def bnt_state():
    return jsonify({
        "capture_disabled": latent_server_waiting,
        "save_w_disabled": backend.generated()
    })

@app.route('/update_style', methods=['POST'])
def update_style():
    data = request.form.to_dict()
    for key, value in data.items():
        if key == 'seed':
            try:
                value = np.int64(int(value))
            except Exception:
                print('update_style: invalid seed')
                return Response(status=500)
            backend.update_seed(value)
        else:
            try:
                value = float(value)
            except Exception:
                print('update_style: non float value')
                return Response(status=500)
            if not backend.update_style(key, value):
                print('update_style: invalid key')
                return Response(status=500)
    return Response(status=204)

@app.route('/capture')
def capture():
    if latent_server_waiting:
        return Response("You shouldn't be here. How you get here?", status=500)
    return jsonify({"success": backend.require_capture()})

@app.route('/generate_styles')
def generate_styles():
    global latent_server_waiting
    with latent_server_cv:
        if not latent_server_waiting:
            print('Post request to latent server')
            try:
                response = requests.post(f'http://{latent_server_url}/receive_image',
                                         files={'image': backend.get_captured_buffer()},
                                         timeout=5)
                if not response.ok:
                    msg = f"Failed to send image to server: {response.status_code}"
                    print(msg)
                    return Response(msg, status=500)
            except requests.exceptions.Timeout:
                msg = "The request timed out"
                print(msg)
                return Response(msg,status=500)
            print('Post success. Waiting for receive...')
            latent_server_waiting = True
        latent_server_cv.wait()
    return Response(status=204)

@app.route('/save_w')
def save_w():
    if not backend.save_w():
        return Response("You shouldn't be here. How you get here?", status=500)
    return Response(status=204)

@app.route('/latent_w', methods=['POST'])
def latent_w():
    global latent_server_waiting
    print('Received from latent server')
    data = request.get_json()
    if 'w' not in data:
        print("No data uploaded")
        return Response("No data uploaded", status=400)

    b64_encoded = data['w']
    try:
        file_bytes = base64.b64decode(b64_encoded)
        w_vector = torch.from_numpy(np.load(io.BytesIO(file_bytes))).to(backend.stylegan.device)
        backend.recv_w(w_vector)
        with latent_server_cv:
            latent_server_waiting = False
            latent_server_cv.notify_all()
        return Response(status=200)
    except Exception as e:
        print(f'When receiving latent w: {e}')
        traceback.print_exc()
        return Response(f"Error processing upload: {str(e)}", status=500)

if __name__ == '__main__':
    print("Starting backend...")
    backend.start()
    print("Backend started")
    print("Setting up server")
    app.run(host=server_ip, port=server_port)
    print("Server stopped")
    backend.stop()
    print("Backend stopped")
