from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from brain_of_the_doctor import encoder_image, analyze_image_with_query
from Voice_of_the_patient import transcribe_with_groq
from Voice_of_the_Doctor import text_to_speech_with_gTTS

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

# Set the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/doctor-response', methods=['POST'])
def doctor_response():
    try:
        # Get the files from the request
        voice_file = request.files['voice']
        image_file = request.files['image']
        
        # Save the files temporarily
        voice_filename = secure_filename(voice_file.filename)
        image_filename = secure_filename(image_file.filename)

        voice_path = os.path.join(app.config['UPLOAD_FOLDER'], voice_filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        voice_file.save(voice_path)
        image_file.save(image_path)

        # Transcribe the voice input
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        audio_text = transcribe_with_groq(voice_path, GOOGLE_API_KEY)
        
        # Encode the image
        image_part = encoder_image(image_path)
        
        # Analyze the image and text
        response_text = analyze_image_with_query(image_part, audio_text)
        
        # Convert the text response to speech
        audio_path = text_to_speech_with_gTTS(response_text)

        # Return the response text and audio path
        return jsonify({
            "responseText": response_text,
            "responseAudio": audio_path
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
