from flask import Flask, render_template, send_from_directory
app = Flask(__name__, static_url_path='', template_folder='views')

@app.route("/lib/<path:path>")
def send_js(path):
    return send_from_directory('lib', path)

@app.route("/data/<path:path>")
def send_data(path):
    return send_from_directory('data', path)

@app.route("/images/<path:path>")
def send_images(path):
    return send_from_directory('images', path)

@app.route("/")
def index():
    return render_template('nbody.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
