# Minimal app.py for testing purposes
from flask import Flask, request, jsonify, render_template_string
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Simple HTML template for testing
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Test App</title></head>
<body>
    <h1>File Upload Test</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
'''

@app.route('/')
def main_page():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check for null bytes (security)
    if '\x00' in file.filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # Check file extension
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Only CSV files allowed.'}), 400
    
    # Save file (in real app, you'd process it)
    if app.config.get('UPLOAD_FOLDER'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
    return jsonify({'message': 'File uploaded successfully', 'filename': file.filename}), 200

if __name__ == '__main__':
    app.run(debug=True)