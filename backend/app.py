from flask import Flask
from flask_cors import CORS
from routes.upload import bp as upload
from routes.chat_blueprint import bp as chat
from routes.file_analysis_endpoint import analysis_bp
from routes.timeseries_chat_blueprint import bp as timeseries_chat_bp
from routes.marketing_chat_blueprint import bp as marketing_chat_bp
from routes.inventory_chat_blueprint import bp as inventory_chat_bp
from routes.supplier_chat_blueprint import bp as supplier_chat_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(upload)
app.register_blueprint(chat)
app.register_blueprint(analysis_bp)
app.register_blueprint(timeseries_chat_bp)
app.register_blueprint(marketing_chat_bp)
app.register_blueprint(inventory_chat_bp)
app.register_blueprint(supplier_chat_bp)

if __name__ == '__main__':
    app.run(debug=True)