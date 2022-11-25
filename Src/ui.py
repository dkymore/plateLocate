from flask import Flask,  request , jsonify
import flask_cors
from pl import workflow

app = Flask(__name__)
flask_cors.CORS(app)

@app.route('/api', methods=['POST'])
def api():
    imgname = request.args.get("name")
    data = request.get_json(silent=True)
    try:
        return jsonify({
            "error":False,
            "imgdata":workflow(imgname,data["img"])
        })
    except Exception as e:
        return jsonify({
            "error":True,
            "message":str(e)
        })

if __name__ == '__main__':
    app.run("0.0.0.0",port=7899)