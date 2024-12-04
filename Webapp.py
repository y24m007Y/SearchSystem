import pandas as pd
import numpy as np
from pathlib2 import Path
from flask import Flask
from flask import url_for
from flask import request
from flask import render_template
from markupsafe import escape

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return do_the_login()
    else:
        return show_the_login_form()
    return 'Index Page'

@app.route('/login')
def login():
    return 'login'

@app.route("/<username>")
def profile(username):
    return f"Hello, {escape(username)}"

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
    print(url_for('static', filename='style.css'))