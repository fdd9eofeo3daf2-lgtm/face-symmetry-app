import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_nose_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
nose_cascade = cv2.CascadeClassifier(_nose_cascade_path) if os.path.exists(_nose_cascade_path) else None
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
_mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
mouth_cascade = cv2.CascadeClassifier(_mouth_cascade_path) if os.path.exists(_mouth_cascade_path) else None

def _get_eyes_center_x(gray_400):
    """両目の中点のX座標（眉間〜鼻根の位置で、中央線として最も安定）。検出できない場合はNone。"""
    eyes = eye_cascade.detectMultiScale(gray_400, 1.1, 6, minSize=(18, 18))
    if len(eyes) < 2:
        return None
    # xでソートし、左端・右端の2つを「左目・右目」として採用（同側の重複検出を排除）
    eyes_sorted = sorted(eyes, key=lambda r: r[0])
    # 左右が十分離れているペアだけ採用（片目が2回検出されるのを避ける）
    min_sep = 60
    left_center = eyes_sorted[0][0] + eyes_sorted[0][2] // 2
    right_center = eyes_sorted[-1][0] + eyes_sorted[-1][2] // 2
    if abs(right_center - left_center) < min_sep:
        return None
    return (left_center + right_center) // 2

def _get_mouth_center_x(gray_400):
    """唇の中央（くぼみ付近）のX座標。口は顔の下側で検出。検出できない場合はNone。"""
    if mouth_cascade is None:
        return None
    h, w = gray_400.shape
    # 口は顔の下半分（y 約45%〜）に限定して検出
    y_start = int(h * 0.45)
    roi = gray_400[y_start:, :]
    mouths = mouth_cascade.detectMultiScale(roi, 1.1, 7, minSize=(25, 15))
    if len(mouths) == 0:
        return None
    # 中央に近い・適度なサイズの口を採用（唇のくぼみ＝上唇中央を想定）
    best = min(mouths, key=lambda r: abs((r[0] + r[2] // 2) - w // 2) + (100 if r[2] * r[3] < 400 or r[2] * r[3] > 8000 else 0))
    center_x = best[0] + best[2] // 2
    return int(np.clip(center_x, 100, 300))

def _get_face_center_x(gray_400):
    """中央線のX座標。両目の中点と唇中央のくぼみを加味してずれを抑える。"""
    eyes_center = _get_eyes_center_x(gray_400)
    mouth_center = _get_mouth_center_x(gray_400)
    if eyes_center is not None and mouth_center is not None:
        # 目と唇の中央をブレンド（眉間〜鼻筋〜人中〜唇が一直線になるように）
        center_x = int(0.55 * eyes_center + 0.45 * mouth_center)
    elif eyes_center is not None:
        center_x = eyes_center
    elif mouth_center is not None:
        center_x = mouth_center
    else:
        if nose_cascade is not None:
            margin = 80
            roi = gray_400[:, margin : 400 - margin]
            noses = nose_cascade.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))
            if len(noses) > 0:
                best = max(noses, key=lambda r: r[2] * r[3])
                center_x = margin + best[0] + best[2] // 2
            else:
                center_x = 200
        else:
            center_x = 200
    return int(np.clip(center_x, 100, 300))

def analyze_skeleton_balanced(face_img, filename):
    canvas = cv2.resize(face_img, (400, 400))
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # 真ん中の定義：両目の中点 ＋ 唇の中央くぼみを加味（眉間〜鼻〜人中〜唇が一直線に）
    center_x = _get_face_center_x(gray)
    half_w = min(center_x, 400 - center_x)
    left_side = gray[:, center_x - half_w : center_x]
    right_side = gray[:, center_x : center_x + half_w]
    right_flipped = cv2.flip(right_side, 1)
    diff = cv2.absdiff(left_side, right_flipped)

    details = []
    total_deduction = 0
    
    # 診断部位の定義（100点満点からの減点方式）
    # 1部位あたりの減点上限（フタ）：12点 → 5項目合計最大60点減、最悪でも40点は残る
    DEDUCTION_CAP_PER_REGION = 12
    COEFFICIENT = 0.15   # 顔の傾き・撮影角を許容し、判定をかなりマイルドに
    IGNORE_DEDUCTION_UNDER = 4  # この値未満は減点0（傾きによる左右差は許容）

    regions = [
        {"range": (50, 110), "name": "眉・額のライン", "label": "Brow", "color": (0, 0, 255)},
        {"range": (110, 160), "name": "目の高さ・形状", "label": "Eye", "color": (50, 50, 255)},
        {"range": (160, 250), "name": "耳・頬の輪郭", "label": "Ear/Cheek", "color": (0, 255, 255)},
        {"range": (250, 330), "name": "口元・あごのライン", "label": "Mouth/Jaw", "color": (0, 165, 255)},
        {"range": (330, 395), "name": "あごの骨格・先端", "label": "Chin", "color": (0, 100, 255)}
    ]

    for reg in regions:
        y_start, y_end = reg["range"]
        reg_diff = np.mean(diff[y_start:y_end, :])
        
        # 減点計算：係数でゆるめ。顔が傾いている／撮影角の差は許容（減点しすぎない）
        deduction = int(reg_diff * COEFFICIENT)
        if deduction < IGNORE_DEDUCTION_UNDER: deduction = 0  # 傾き・撮影角による差は許容
        deduction = min(deduction, DEDUCTION_CAP_PER_REGION)  # 1部位の減点フタ（12点まで）
        
        # 理由とステータスの判定
        if deduction >= 12:
            reason = "顕著な骨格のズレを確認"
            status = "critical"
            y_mid = (y_start + y_end) // 2
            cv2.arrowedLine(canvas, (380, y_mid), (330, y_mid), reg["color"], 2, tipLength=0.3)
            cv2.putText(canvas, f"-{deduction}", (340, y_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, reg["color"], 1)
        elif deduction >= 5:
            reason = "筋肉や脂肪の付き方に左右差"
            status = "warning"
        else:
            reason = "非常に高い対称性です"
            status = "good"
            
        total_deduction += deduction
        details.append({
            "name": reg["name"], 
            "deduction": deduction, 
            "reason": reason,
            "status": status
        })

    # 中央ガイドライン（両目の中点＋唇中央を加味）
    cv2.line(canvas, (center_x, 0), (center_x, 400), (255, 255, 255), 1)

    result_filename = "final_" + filename
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), canvas)
    
    # 100点から減点分を引く
    final_score = 100 - total_deduction
    final_score = max(0, min(100, final_score))
    
    return final_score, result_filename, details

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            if img is None: return render_template("index.html", answer="画像エラー")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = img[max(0, y-30):min(img.shape[0], y+h+30), max(0, x-30):min(img.shape[1], x+w+30)]
                score, res_img, details = analyze_skeleton_balanced(face_roi, filename)
                return render_template("index.html", score=score, result_img=res_img, details=details)
            else:
                return render_template("index.html", answer="顔が検出されませんでした。")
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)