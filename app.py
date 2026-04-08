from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_engine.predictor import analyze_credit_application

app = Flask(__name__)
# Enable CORS biar Frontend React/Vue bisa nembak API ini dengan lancar
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Welcome to Credit Union AI MVP API by Engelbertus",
        "version": "1.0.0"
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Menangkap data JSON dari body request frontend
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Data JSON tidak ditemukan di *body request*!"}), 400
            
        # Eksekusi AI Core Logic
        result = analyze_credit_application(data)
        
        if "error" in result:
             return jsonify(result), 500
             
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Running server di port 5000, debug mode ON buat MVP
    print("🚀 Memulai Flask Server *Credit Scoring API*...")
    app.run(host='0.0.0.0', port=5000, debug=True)