# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Configure Gemini API
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyCKGqr2ctSxDcrFgr7fgzhAM8vQF-ZuAnM")
# Initialize the model
model = genai.GenerativeModel('gemini-pro')

def get_bail_application(lawyer_name: str, input_text: str, n_words: int,theme:str) -> str:
    try:
        app.logger.info("Initializing Gemini LLM model...")
        start_time = time.time()

        prompt = f"""
        [Your Name] = {input_text}

[Phone Number]
7/9/24

Honorable Judge Harshit
Kolkata High Court
Kolkata -70002
        "write a bail application.
        name of lawyer will be {lawyer_name}
        name of client will be {input_text}
        crime will be {theme}
        number of word in the bail application will be {n_words} words.
        city will be kolkata
        and judge name will be raghav jain 

        
        """

        app.logger.info("Generating bail application...")
        
        response = model.generate_content(prompt)

        app.logger.info(f"Bail application generated in {time.time() - start_time:.2f} seconds")

        return response.text

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return None

@app.route('/api/generate_bail_application', methods=['POST'])
def generate_bail_application():
    try:
        data = request.json
        lawyer_name = data.get('lawyer_name')
        input_text = data.get('input_text')
        n_words = data.get('n_words', 200)
        theme =data.get('theme')

        if not lawyer_name or not input_text:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        if not isinstance(n_words, int) or n_words <= 0:
            return jsonify({"success": False, "error": "Invalid n_words value"}), 400

        response = get_bail_application(lawyer_name, input_text, n_words,theme)

        if response:
            return jsonify({"success": True, "bail_application": response})
        else:
            return jsonify({"success": False, "error": "Failed to generate bail application"}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)