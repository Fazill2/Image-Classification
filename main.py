from flask import Flask

from api.routes import api_bp
from data_loading.data_loader import DataLoader


app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')

def main():
    data_loader = DataLoader('data')
    data_loader.load()

if __name__ == '__main__':
    main()
    app.run(debug=True)
