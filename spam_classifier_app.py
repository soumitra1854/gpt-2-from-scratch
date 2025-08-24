from flask import Flask, render_template_string, request, jsonify
import torch
import tiktoken
import time
import csv
import io
from dataloader import create_spam_dataloader
from model import GPTSpamClassification
from utils import GPT2_CONFIG_124M

app = Flask(__name__)

class SpamClassifier:
    def __init__(self, model_path="checkpoints/finetuned_spam_model_only.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model_path = model_path
        self.model = None
        try:
            # Load the training dataloader to determine the max_length the model was trained on
            train_loader = create_spam_dataloader("data/train.csv", shuffle=False)
            self.max_length = train_loader.dataset.max_length
        except FileNotFoundError:
            print("Error: train.csv not found. Please run prepare_data.py first.")
            exit()

        # Statistics tracking
        self.stats = {
            'total_analyzed': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'total_confidence': 0.0,
            'total_processing_time': 0.0
        }
        
        # Load the model
        self._load_model()
        print(f"Spam Classifier initialized on device: {self.device}")

    def _load_model(self):
        """Load the fine-tuned spam classification model"""
        try:
            self.model = GPTSpamClassification(GPT2_CONFIG_124M).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded spam classification model from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file not found at '{self.model_path}'")
            print("Please run `python fine_tune.py --classification_ft` to train the model first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def classify_text(self, text):
        """Classify a single piece of text as spam or ham"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        try:
            # Prepare the input
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]  # Truncate if too long
            
            # Use proper padding token
            pad_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
            padded_tokens = torch.full((self.max_length,), pad_token_id, dtype=torch.long)
            padded_tokens[:len(token_ids)] = torch.tensor(token_ids)
            
            input_tensor = padded_tokens.unsqueeze(0).to(self.device)
            
            # Get the model's prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            predicted_label = torch.argmax(logits[:, -1, :], dim=-1).item()
            confidence = probs[0, predicted_label].item()
            
            classification = "Spam" if predicted_label == 1 else "Ham"
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_analyzed'] += 1
            self.stats['total_confidence'] += confidence
            self.stats['total_processing_time'] += processing_time
            
            if classification == "Spam":
                self.stats['spam_detected'] += 1
            else:
                self.stats['ham_detected'] += 1
            
            return {
                "classification": classification,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return {"error": "Classification failed"}

    def classify_batch(self, messages):
        """Classify a batch of messages"""
        results = []
        for message in messages:
            if message.strip():
                result = self.classify_text(message.strip())
                if "error" not in result:
                    results.append({
                        "message": message.strip(),
                        **result
                    })
        return results

    def get_stats(self):
        """Get current statistics"""
        if self.stats['total_analyzed'] == 0:
            return {
                "total_analyzed": 0,
                "spam_percentage": 0,
                "ham_percentage": 0,
                "average_confidence": 0,
                "average_processing_time": 0
            }
        
        return {
            "total_analyzed": self.stats['total_analyzed'],
            "spam_detected": self.stats['spam_detected'],
            "ham_detected": self.stats['ham_detected'],
            "spam_percentage": (self.stats['spam_detected'] / self.stats['total_analyzed']) * 100,
            "ham_percentage": (self.stats['ham_detected'] / self.stats['total_analyzed']) * 100,
            "average_confidence": (self.stats['total_confidence'] / self.stats['total_analyzed']) * 100,
            "average_processing_time": self.stats['total_processing_time'] / self.stats['total_analyzed']
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_analyzed': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'total_confidence': 0.0,
            'total_processing_time': 0.0
        }

# Initialize the spam classifier
spam_classifier = SpamClassifier()

# Routes
@app.route('/')
def index():
    """Serve the main spam classifier interface"""
    try:
        with open('templates/spam_classifier_app.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to basic template if HTML file not found
        return '''
        <html>
        <head><title>Spam Classifier</title></head>
        <body>
            <h1>Spam Classifier</h1>
            <p>Please ensure spam_classifier.html is in the same directory as this Python file.</p>
            <p>Or create the HTML file from the provided template.</p>
        </body>
        </html>
        '''

@app.route('/classify', methods=['POST'])
def classify_message():
    """Classify a single message"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if len(message) > 10000:  # Limit message length
            return jsonify({'error': 'Message too long (max 10,000 characters)'}), 400
        
        result = spam_classifier.classify_text(message)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify({
            'classification': result['classification'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in classify endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """Classify multiple messages from file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (10MB limit)
        if request.content_length > 10 * 1024 * 1024:
            return jsonify({'error': 'File size too large (max 10MB)'}), 400
        
        # Read file content
        file_content = file.read().decode('utf-8')
        
        # Parse based on file type
        messages = []
        if file.filename.endswith('.csv'):
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(file_content))
            for row in csv_reader:
                if row:  # Skip empty rows
                    messages.append(row[0])  # Take first column
        else:
            # Parse as text file (one message per line)
            messages = file_content.strip().split('\n')
        
        # Limit number of messages to prevent overload
        if len(messages) > 1000:
            messages = messages[:1000]
        
        # Filter out empty messages
        messages = [msg.strip() for msg in messages if msg.strip()]
        
        if not messages:
            return jsonify({'error': 'No valid messages found in file'}), 400
        
        # Classify all messages
        results = spam_classifier.classify_batch(messages)
        
        # Calculate summary statistics
        spam_count = sum(1 for r in results if r['classification'] == 'Spam')
        ham_count = len(results) - spam_count
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_percentage': (spam_count / len(results)) * 100 if results else 0,
                'average_confidence': avg_confidence * 100
            },
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in batch classify endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get current classification statistics"""
    try:
        stats = spam_classifier.get_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in stats endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/reset_stats', methods=['POST'])
def reset_statistics():
    """Reset classification statistics"""
    try:
        spam_classifier.reset_stats()
        return jsonify({'status': 'success', 'message': 'Statistics reset'})
    except Exception as e:
        print(f"Error in reset stats endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        return jsonify({
            'model_loaded': spam_classifier.model is not None,
            'device': str(spam_classifier.device),
            'max_length': spam_classifier.max_length,
            'model_path': spam_classifier.model_path,
            'vocab_size': GPT2_CONFIG_124M['vocab_size'],
            'context_length': GPT2_CONFIG_124M['context_length']
        })
    except Exception as e:
        print(f"Error in model info endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': spam_classifier.model is not None,
        'device': str(spam_classifier.device)
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Spam Classifier Web App...")
    print(f"Device: {spam_classifier.device}")
    print(f"Model loaded: {spam_classifier.model is not None}")
    print("Navigate to http://localhost:5001 to use the spam classifier")
    
    # Check if model is loaded
    if spam_classifier.model is None:
        print("\n⚠️  WARNING: Model not loaded!")
        print("Please run `python fine_tune.py --classification_ft` to train the model first.")
        print("The web app will still start but classification won't work until the model is loaded.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)