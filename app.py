from application import Application
import flask
import datetime
import gc
import logging, cfg, time
from waitress import serve
from data_processing_unit.utils import *

app = flask.Flask(__name__)
obj_app = Application()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/maskrcnntests3", methods=["POST"])
def getdatatightfits3path():
    if flask.request.method == "POST":
        status = False
        try:
            data = flask.request.json
            imagefrontpath = str(data['imagefront'])
            imagesidepath = str(data['imageside'])
            front,side = gets3images(imagefrontpath,imagesidepath)

            front, side, status = obj_app.dataProcessingUnit(front,side)

            return flask.jsonify(status)
        except:
            return flask.jsonify(status)




if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5001)
