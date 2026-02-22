import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
import numpy as np
from flask import Flask, render_template, request, send_from_directory
import os
import uuid
import urllib.request

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- モデルのダウンロード＆初期化 ---
MODEL_PATH = '/tmp/face_landmarker.task'
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

_landmarker = None

def get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def _status(penalty, crit_thresh=8):
    if penalty == 0:
        return 'good'
    elif penalty < crit_thresh:
        return 'warning'
    else:
        return 'critical'


def calculate_score(landmarks):
    face_width = abs(landmarks[454].x - landmarks[234].x)
    details = []

    # --- 1. 眉・額のライン ---
    eyebrow_l_y = (landmarks[105].y + landmarks[66].y + landmarks[107].y) / 3
    eyebrow_r_y = (landmarks[334].y + landmarks[296].y + landmarks[336].y) / 3
    eyebrow_diff_pct = (abs(eyebrow_l_y - eyebrow_r_y) / face_width) * 100
    penalty_eyebrow = round(max(0, (eyebrow_diff_pct - 1.0) * 12), 1)

    if penalty_eyebrow == 0:
        eyebrow_reason = f"左右の眉の高さはほぼ均等です（差 {eyebrow_diff_pct:.2f}%）"
    elif penalty_eyebrow < 8:
        eyebrow_reason = f"眉の高さに軽度の左右差があります（差 {eyebrow_diff_pct:.2f}%）"
    else:
        eyebrow_reason = f"眉の高さに顕著な左右差が検出されました（差 {eyebrow_diff_pct:.2f}%）"

    details.append({'name': '眉・額のライン', 'deduction': penalty_eyebrow,
                    'reason': eyebrow_reason, 'status': _status(penalty_eyebrow)})

    # --- 2. 目の高さ・形状 ---
    eye_l_y = (landmarks[33].y + landmarks[133].y) / 2
    eye_r_y = (landmarks[263].y + landmarks[362].y) / 2
    eye_diff_pct = (abs(eye_l_y - eye_r_y) / face_width) * 100
    penalty_eye = round(max(0, (eye_diff_pct - 0.4) * 15), 1)

    if penalty_eye == 0:
        eye_reason = f"左右の目の高さはほぼ均等です（差 {eye_diff_pct:.2f}%）"
    elif penalty_eye < 8:
        eye_reason = f"目の高さに軽度の左右差があります（差 {eye_diff_pct:.2f}%）"
    else:
        eye_reason = f"目の高さに顕著な左右差が検出されました（差 {eye_diff_pct:.2f}%）"

    details.append({'name': '目の高さ・形状', 'deduction': penalty_eye,
                    'reason': eye_reason, 'status': _status(penalty_eye)})

    # --- 3. 耳・頬の輪郭 ---
    center_x = landmarks[4].x
    dist_l = abs(center_x - landmarks[234].x)
    dist_r = abs(center_x - landmarks[454].x)
    contour_diff_pct = (abs(dist_l - dist_r) / face_width) * 100
    penalty_contour = round(max(0, (contour_diff_pct - 2.0) * 8), 1)

    if penalty_contour == 0:
        contour_reason = f"頬・耳のラインの左右バランスは良好です（差 {contour_diff_pct:.2f}%）"
    elif penalty_contour < 8:
        contour_reason = f"頬のラインにわずかな左右差があります（差 {contour_diff_pct:.2f}%）"
    else:
        contour_reason = f"頬・耳のラインに明確な左右差が見られます（差 {contour_diff_pct:.2f}%）"

    details.append({'name': '耳・頬の輪郭', 'deduction': penalty_contour,
                    'reason': contour_reason, 'status': _status(penalty_contour)})

    # --- 4. 口元・あごのライン ---
    mouth_diff_pct = (abs(landmarks[61].y - landmarks[291].y) / face_width) * 100
    penalty_mouth = round(max(0, (mouth_diff_pct - 1.0) * 10), 1)

    if penalty_mouth == 0:
        mouth_reason = f"左右の口角の高さはほぼ均等です（差 {mouth_diff_pct:.2f}%）"
    elif penalty_mouth < 8:
        mouth_reason = f"口角に軽度の左右差があります（差 {mouth_diff_pct:.2f}%）"
    else:
        mouth_reason = f"口角に顕著な左右差が検出されました（差 {mouth_diff_pct:.2f}%）"

    details.append({'name': '口元・あごのライン', 'deduction': penalty_mouth,
                    'reason': mouth_reason, 'status': _status(penalty_mouth)})

    # --- 5. あごの骨格・先端 ---
    chin_dev_pct = (abs(landmarks[152].x - landmarks[4].x) / face_width) * 100
    dist_l_jaw = abs(landmarks[172].y - landmarks[152].y)
    dist_r_jaw = abs(landmarks[397].y - landmarks[152].y)
    jaw_sym_pct = (abs(dist_l_jaw - dist_r_jaw) / face_width) * 100
    penalty_jaw = round(
        max(0, (chin_dev_pct - 1.5) * 10) + max(0, (jaw_sym_pct - 2.0) * 6), 1
    )

    if penalty_jaw == 0:
        jaw_reason = f"あごの位置・骨格は左右均等です（中心ズレ {chin_dev_pct:.2f}%）"
    elif penalty_jaw < 8:
        jaw_reason = f"あごにわずかな偏りが見られます（中心ズレ {chin_dev_pct:.2f}%）"
    else:
        jaw_reason = f"あごの位置に顕著な左右偏位が検出されました（中心ズレ {chin_dev_pct:.2f}%）"

    details.append({'name': 'あごの骨格・先端', 'deduction': penalty_jaw,
                    'reason': jaw_reason, 'status': _status(penalty_jaw)})

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = get_landmarker().detect(mp_image)

        if not result.face_landmarks:
            return render_template('index.html', score=None, details=[],
                                   result_img=None, error='顔が検出できませんでした')

        landmarks = result.face_landmarks[0]
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
