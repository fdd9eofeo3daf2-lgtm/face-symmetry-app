import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, send_from_directory
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def _status(penalty, warn_thresh=3, crit_thresh=8):
    if penalty == 0:
        return 'good'
    elif penalty < crit_thresh:
        return 'warning'
    else:
        return 'critical'


def calculate_score(landmarks):
    face_width = abs(landmarks[454].x - landmarks[234].x)
    details = []

    # --- 1. 目の高さの左右対称 ---
    eye_diff = abs(landmarks[33].y - landmarks[263].y)
    eye_diff_pct = (eye_diff / face_width) * 100
    penalty_eye = round(max(0, (eye_diff_pct - 0.4) * 15), 1)

    if penalty_eye == 0:
        eye_reason = f"左右の目の高さはほぼ均等です（差 {eye_diff_pct:.2f}%）"
    elif penalty_eye < 8:
        eye_reason = f"目の高さに軽度の左右差があります（差 {eye_diff_pct:.2f}%）"
    else:
        eye_reason = f"目の高さに顕著な左右差が検出されました（差 {eye_diff_pct:.2f}%）"

    details.append({'name': '目の対称性', 'deduction': penalty_eye,
                    'reason': eye_reason, 'status': _status(penalty_eye)})

    # --- 2. 口角の高さの左右対称 ---
    mouth_diff = abs(landmarks[61].y - landmarks[291].y)
    mouth_diff_pct = (mouth_diff / face_width) * 100
    penalty_mouth = round(max(0, (mouth_diff_pct - 1.0) * 10), 1)

    if penalty_mouth == 0:
        mouth_reason = f"左右の口角の高さはほぼ均等です（差 {mouth_diff_pct:.2f}%）"
    elif penalty_mouth < 8:
        mouth_reason = f"口角に軽度の左右差があります（差 {mouth_diff_pct:.2f}%）"
    else:
        mouth_reason = f"口角に顕著な左右差が検出されました（差 {mouth_diff_pct:.2f}%）"

    details.append({'name': '口角の対称性', 'deduction': penalty_mouth,
                    'reason': mouth_reason, 'status': _status(penalty_mouth)})

    # --- 3. 顔の輪郭の左右差 ---
    center_x = landmarks[4].x
    dist_l = abs(center_x - landmarks[234].x)
    dist_r = abs(center_x - landmarks[454].x)
    contour_diff_pct = (abs(dist_l - dist_r) / face_width) * 100
    penalty_contour = round(max(0, (contour_diff_pct - 2.0) * 8), 1)

    if penalty_contour == 0:
        contour_reason = f"顔の輪郭の左右バランスは良好です（差 {contour_diff_pct:.2f}%）"
    elif penalty_contour < 8:
        contour_reason = f"顔の輪郭にわずかな左右差があります（差 {contour_diff_pct:.2f}%）"
    else:
        contour_reason = f"顔の輪郭に明確な左右差が見られます（差 {contour_diff_pct:.2f}%）"

    details.append({'name': '顔輪郭の対称性', 'deduction': penalty_contour,
                    'reason': contour_reason, 'status': _status(penalty_contour)})

    # --- 4. 縦の黄金比（1:1:1）---
    top_h = abs(landmarks[10].y - landmarks[168].y)
    mid_h = abs(landmarks[168].y - landmarks[2].y)
    btm_h = abs(landmarks[2].y - landmarks[152].y)
    avg_h = (top_h + mid_h + btm_h) / 3

    penalty_ratio_v = 0
    for h in [top_h, mid_h, btm_h]:
        diff_pct = (abs(h - avg_h) / face_width) * 100
        if diff_pct > 3.0:
            penalty_ratio_v += (diff_pct - 3.0) * 5
    penalty_ratio_v = round(penalty_ratio_v, 1)

    top_r = round(top_h / avg_h * 100)
    mid_r = round(mid_h / avg_h * 100)
    btm_r = round(btm_h / avg_h * 100)

    if penalty_ratio_v == 0:
        ratio_v_reason = f"額：中顔面：下顔面 ≈ {top_r}:{mid_r}:{btm_r} — 理想的な縦三等分です"
    elif penalty_ratio_v < 8:
        ratio_v_reason = f"額：中顔面：下顔面 = {top_r}:{mid_r}:{btm_r} — 縦比率にわずかなズレがあります"
    else:
        ratio_v_reason = f"額：中顔面：下顔面 = {top_r}:{mid_r}:{btm_r} — 縦の三等分比率から大きく外れています"

    details.append({'name': '縦の黄金比（1:1:1）', 'deduction': penalty_ratio_v,
                    'reason': ratio_v_reason, 'status': _status(penalty_ratio_v)})

    # --- 5. 下顔面の黄金比（鼻下〜唇 1 : 唇〜あご 2）---
    philtrum = abs(landmarks[2].y - landmarks[0].y)
    chin = abs(landmarks[0].y - landmarks[152].y)
    actual_ratio = round(chin / philtrum, 2) if philtrum > 0 else 0
    penalty_ratio_lower = 5 if not (1.8 <= actual_ratio <= 2.2) else 0

    if penalty_ratio_lower == 0:
        ratio_lower_reason = f"鼻下〜唇：唇〜あご = 1:{actual_ratio} — 理想比率 1:2 に合致しています"
    else:
        ratio_lower_reason = f"鼻下〜唇：唇〜あご = 1:{actual_ratio} — 理想比率 1:2 からズレがあります"

    details.append({'name': '下顔面の黄金比（1:2）', 'deduction': penalty_ratio_lower,
                    'reason': ratio_lower_reason, 'status': _status(penalty_ratio_lower, warn_thresh=1)})

    score = round(max(0, min(100, 100 - sum(d['deduction'] for d in details))), 1)
    return score, details


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', score=None, details=[],
                                   result_img=None, error='ファイルが選択されていません')

        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return render_template('index.html', score=None, details=[],
                                   result_img=None, error='画像を読み込めませんでした')

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            return render_template('index.html', score=None, details=[],
                                   result_img=None, error='顔が検出できませんでした')

        landmarks = results.multi_face_landmarks[0].landmark
        score, details = calculate_score(landmarks)

        filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), img)

        return render_template('index.html', score=score, details=details,
                               result_img=filename, error=None)

    return render_template('index.html', score=None, details=[], result_img=None, error=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
