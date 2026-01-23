from flask import Flask, request, render_template_string, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
from PIL import Image
import io
import base64

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# تحميل الموديل
model = load_model("mammals_mobilenet.h5")

# أسماء الـ 45 صنف
class_names = [
    "african_elephant","alpaca","american_bison","anteater","arctic_fox",
    "armadillo","baboon","badger","blue_whale","brown_bear","camel","dolphin",
    "giraffe","groundhog","highland_cattle","horse","jackal","kangaroo","koala",
    "manatee","mongoose","mountain_goat","opossum","orangutan","otter","polar_bear",
    "porcupine","red_panda","rhinoceros","sea_lion","seal","snow_leopard","squirrel",
    "sugar_glider","tapir","vampire_bat","vicuna","walrus","warthog","water_buffalo",
    "weasel","wildebeest","wombat","yak","zebra"
]

# HTML للواجهة
html_template = """<!doctype html>
<html>
<head>
  <title>🐾 Mammals Classifier</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(to right,#f0f4f8,#ffffff); display:flex; flex-direction:column; align-items:center; padding-top:40px;}
    h2 { color:#004d40; }
    form { background:white; padding:20px; border-radius:15px; box-shadow:0 8px 20px rgba(0,0,0,0.1); text-align:center;}
    input[type="file"], button { margin-bottom:15px; padding:10px; font-size:16px; border-radius:8px; cursor:pointer;}
    button { background-color:#00796b; color:white; border:none;}
    button:hover { background-color:#004d40; }
    h3 { color:#004d40; }
    img { margin-top:20px; max-width:400px; border-radius:10px; box-shadow:0 5px 15px rgba(0,0,0,0.1);}
  </style>
</head>
<body>
  <h2>📷 ارفع صورة أو استخدم الكاميرا لتصنيف الثدييات</h2>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <br>
    <input type=submit value="رفع الصورة وتصنيفها">
  </form>
  <h3>أو استخدم الكاميرا</h3>
  <video id="video" autoplay width="400"></video><br>
  <button id="snap">التقاط وتصنيف</button>
  <canvas id="canvas" style="display:none;"></canvas>
  {% if prediction %}
    <h3>✅ Top 3 التصنيفات:</h3>
    <ul>
    {% for cls, conf in prediction %}
      <li>{{ cls }} : {{ (conf*100)|round(2) }}%</li>
    {% endfor %}
    </ul>
    <img src="{{ image_url }}" alt="الصورة المرفوعة">
  {% endif %}
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => { video.srcObject = stream; }).catch(err => { console.error(err); });
document.getElementById('snap').addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/png');
    fetch("/", { method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:"webcam_image="+encodeURIComponent(dataURL) }).then(()=>{ location.reload(); });
});
</script>
</body>
</html>"""

# تجهيز الصورة
def prepare_image(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = image.load_img(img_path_or_pil, target_size=(224,224))
    else:
        img = img_path_or_pil.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    image_url = None
    if request.method=="POST":
        if "file" in request.files and request.files['file'].filename != "":
            file = request.files['file']
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("static", filename)
            file.save(filepath)
            img = prepare_image(filepath)
            preds = model.predict(img)[0]
            top3_idx = preds.argsort()[-3:][::-1]
            prediction = [(class_names[i], preds[i]) for i in top3_idx]
            image_url = f"/static/{filename}"
        elif "webcam_image" in request.form:
            data = request.form["webcam_image"].split(",")[1]
            img_bytes = io.BytesIO(base64.b64decode(data))
            img = Image.open(img_bytes)
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("static", filename)
            img.save(filepath)
            img_array = prepare_image(img)
            preds = model.predict(img_array)[0]
            top3_idx = preds.argsort()[-3:][::-1]
            prediction = [(class_names[i], preds[i]) for i in top3_idx]
            image_url = f"/static/{filename}"
    return render_template_string(html_template, prediction=prediction, image_url=image_url)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
