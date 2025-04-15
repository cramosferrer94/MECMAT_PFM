# app.py
from flask import Flask, render_template
from app.routes import main

app = Flask(__name__, template_folder="templates")  # Ajusta la carpeta de plantillas según corresponda
app.register_blueprint(main)

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
