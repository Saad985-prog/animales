
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os, uuid, base64, io
from PIL import Image

app = Flask(__name__)

model = load_model("mammals_mobilenet.h5")

class_names = [
"african_elephant","alpaca","american_bison","anteater","arctic_fox",
"armadillo","baboon","badger","blue_whale","brown_bear","camel","dolphin",
"giraffe","groundhog","highland_cattle","horse","jackal","kangaroo","koala",
"manatee","mongoose","mountain_goat","opossum","orangutan","otter","polar_bear",
"porcupine","red_panda","rhinoceros","sea_lion","seal","snow_leopard","squirrel",
"sugar_glider","tapir","vampire_bat","vicuna","walrus","warthog","water_buffalo",
"weasel","wildebeest","wombat","yak","zebra"
]

os.makedirs("static", exist_ok=True)

def prepare_image(img):
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

@app.route("/", methods=["GET","POST"])
def index():
    prediction, image_url = None, None

    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            f = request.files["file"]
            name = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join("static", name)
            f.save(path)
            img = Image.open(path)
            preds = model.predict(prepare_image(img))[0]

        elif "webcam_image" in request.form:
            data = request.form["webcam_image"].split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(data)))
            name = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join("static", name)
            img.save(path)
            preds = model.predict(prepare_image(img))[0]

        else:
            return render_template("index.html")

        top3 = preds.argsort()[-3:][::-1]
        prediction = [(class_names[i], float(preds[i])) for i in top3]
        image_url = f"/static/{name}"

    return render_template("index.html", prediction=prediction, image_url=image_url)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
