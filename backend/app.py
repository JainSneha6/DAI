from flask import Flask
from flask_cors import CORS
from routes.upload import bp as upload

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(upload)

if __name__ == '__main__':
    app.run(debug=True)