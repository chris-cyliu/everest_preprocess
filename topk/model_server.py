import config as cfg
from flask import Flask
from models.models import YOLOv3

app = Flask("mdn_server")
model = YOLOv3(cfg)


@app.route('/infer/<string:image_path>', methods=['GET'])
def infer(image_path):
    # call mdn model here
    score = len(model.predict(image_path)[0])
    print("receiving request for: " + image_path)
    return str(score)


if __name__ == "__main__":
    app.run()
