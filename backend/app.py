from flask import Flask
from flask_cors import CORS
from routes.upload import bp as upload
from routes.chat_blueprint import bp as chat

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(upload)
app.register_blueprint(chat)

if __name__ == '__main__':
    app.run(debug=True)