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
def get_bail_application(lawyer_name: str, input_text: str, n_words: int) -> str:
    try:
        app.logger.info("Initializing Gemini LLM model...")
        start_time = time.time()

        template = """
        Write a bail letter addressed to a judge explaining why {lawyer_name}'s client should be granted bail for {input_text}.
        Letter should be within {n_words} words.
        """

        prompt = template.format(
            lawyer_name=lawyer_name,
            input_text=input_text,
            n_words=n_words
        )

        app.logger.info("Generating bail application...")
        
        result = genai.generate_text(
            model="models/chat-bison-001",
            prompt=prompt,
            temperature=0.1,
            max_output_tokens=1024
        )

        app.logger.info(f"Bail application generated in {time.time() - start_time:.2f} seconds")

        return result.result

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

        if not lawyer_name or not input_text:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        if not isinstance(n_words, int) or n_words <= 0:
            return jsonify({"success": False, "error": "Invalid n_words value"}), 400

        response = get_bail_application(lawyer_name, input_text, n_words)

        if response:
            return jsonify({"success": True, "bail_application": response})
        else:
            return jsonify({"success": False, "error": "Failed to generate bail application"}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
