from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    iframe = 'http://localhost:8001/index.xhtml'
    return render_template('index.html', iframe=iframe)

@app.route('/annotation_begin')
def annotation_begin():
    return render_template('annotation_begin.html')

@app.route('/annotation_end')
def annotation_end():
    return render_template('annotation_end.html')

if __name__ == '__main__':
    app.run()
