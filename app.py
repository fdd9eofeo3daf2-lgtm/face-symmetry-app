import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# MediaPipeの設定
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def calculate_score(landmarks):
    # 顔の両端（234番と454番）で顔の横幅を取得
    face_width = abs(landmarks[454].x - landmarks[234].x)
    
    # --- 1. 左右対称（シンメトリー）チェック ---
    # 目の高さの差（左:33, 右:263）
    eye_diff = abs(landmarks[33].y - landmarks[263].y)
    eye_diff_percent = (eye_diff / face_width) * 100
    
    # 口角の高さの差（左:61, 右:291）
    mouth_diff = abs(landmarks[61].y - landmarks[291].y)
    mouth_diff_percent = (mouth_diff / face_width) * 100
    
    # 減点計算：1%（遊び）を超えたら1%につき10点引く
    penalty_sym = 0
    if eye_diff_percent > 1.0:
        penalty_sym += (eye_diff_percent - 1.0) * 10
    if mouth_diff_percent > 1.0:
        penalty_sym += (mouth_diff_percent - 1.0) * 10

    # --- 2. 縦の黄金比（1:1:1）チェック ---
    # 生え際(10), 眉間(168), 鼻下(2), あご先(152)
    top_h = abs(landmarks[10].y - landmarks[168].y)
    mid_h = abs(landmarks[168].y - landmarks[2].y)
    btm_h = abs(landmarks[2].y - landmarks[152].y)
    
    avg_h = (top_h + mid_h + btm_h) / 3
    penalty_ratio_v = 0
    for h in [top_h, mid_h, btm_h]:
        diff_percent = (abs(h - avg_h) / face_width) * 100
        if diff_percent > 3.0: # 3%までは誤差
            penalty_ratio_v += (diff_percent - 3.0) * 5

    # --- 3. 下顔面の黄金比（鼻下〜唇 1 : 唇〜あご 2） ---
    # 鼻下(2), 唇中央(0), あご先(152)
    philtrum = abs(landmarks[2].y - landmarks[0].y)
    chin = abs(landmarks[0].y - landmarks[152].y)
    
    # 理想は chin / philtrum = 2.0
    actual_ratio = chin / philtrum if philtrum != 0 else 0
    penalty_ratio_lower = 0
    if not (1.8 <= actual_ratio <= 2.2):
        penalty_ratio_lower = 5 # 理想から外れたら一律5点減点

    # 最終スコア算出
    score = 100 - (penalty_sym + penalty_ratio_v + penalty_ratio_lower)
    return round(max(0, min(100, score)), 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        score = calculate_score(landmarks)
        return jsonify({'score': score})
    
    return jsonify({'error': '顔が検出できませんでした'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)