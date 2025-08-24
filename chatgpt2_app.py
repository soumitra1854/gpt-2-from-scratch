from flask import Flask, render_template, request, jsonify, session
import torch
import tiktoken
from model import GPTModel
from utils import (
    generate,
    GPT2_CONFIG_124M,
    Custom_GPT2_config,
    text_to_token_ids,
    token_ids_to_text,
    load_weights_from_gpt2
)
import uuid
import threading
import time

app = Flask(__name__)
app.secret_key = 'how-you-doin'


class ChatBot:
    def __init__(self, model_path="checkpoints/trained_model_model_only.pth"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model_path = model_path

        # Initialize with custom model by default
        self.current_model_type = "custom"
        self.model = None
        self.config = None

        # Load the default model
        self._load_custom_model()
        print(f"ChatBot initialized on device: {self.device}")

    def _load_custom_model(self):
        """Load the custom trained model"""
        try:
            self.model = GPTModel(Custom_GPT2_config).to(self.device)
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
            self.config = Custom_GPT2_config
            self.current_model_type = "custom"
            self.model.eval()
            print(f"Loaded custom model from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Custom model file not found at '{self.model_path}'")
            return False
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return False

    def _load_openai_model(self):
        """Load the OpenAI GPT-2 model"""
        try:
            print("Loading pre-trained OpenAI GPT-2 weights...")
            self.model = GPTModel(GPT2_CONFIG_124M).to(self.device)
            self.model = load_weights_from_gpt2(self.model)
            self.config = GPT2_CONFIG_124M
            self.current_model_type = "openai"
            self.model.eval()
            print("OpenAI GPT-2 model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading OpenAI model: {e}")
            return False

    def switch_model(self, model_type):
        """Switch between custom and OpenAI models"""
        if model_type == self.current_model_type:
            return {"success": True, "message": f"Already using {model_type} model"}

        # Clear GPU memory before loading new model
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if model_type == "openai":
            success = self._load_openai_model()
            if success:
                return {"success": True, "message": "Switched to OpenAI GPT-2 model"}
            else:
                # Fallback to custom model
                self._load_custom_model()
                return {"success": False, "message": "Failed to load OpenAI model, reverted to custom model"}

        elif model_type == "custom":
            success = self._load_custom_model()
            if success:
                return {"success": True, "message": "Switched to custom GPT-2 model"}
            else:
                # Fallback to OpenAI model
                self._load_openai_model()
                return {"success": False, "message": "Failed to load custom model, switched to OpenAI model"}

        return {"success": False, "message": "Invalid model type"}

    def get_current_model_info(self):
        """Get information about the currently loaded model"""
        return {
            "type": self.current_model_type,
            "context_length": self.config["context_length"] if self.config else 0,
            "vocab_size": self.config["vocab_size"] if self.config else 0
        }

    def generate_response(self, prompt, max_tokens=100, temperature=0.8, top_k=25):
        """Generate response from the model"""
        try:
            # Encode the prompt
            encoded_prompt = text_to_token_ids(
                prompt, self.tokenizer).to(self.device)

            with torch.no_grad():
                output_ids = generate(
                    model=self.model,
                    input_ids=encoded_prompt,
                    max_new_tokens=max_tokens,
                    context_size=self.config["context_length"],
                    temperature=temperature,
                    top_k=top_k
                )

            # Decode the response
            result = token_ids_to_text(output_ids, self.tokenizer)

            return result if result else "I'm not sure how to respond to that."

        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating a response."


# Initialize the chatbot
chatbot = ChatBot(model_path="checkpoints/trained_model_model_only.pth")


@app.route('/')
def index():
    """Serve the main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = []
    return render_template('chatgpt2_app.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Get generation parameters from request or use defaults
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 25)

        # Generate response
        bot_response = chatbot.generate_response(
            prompt=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Store in session history
        if 'chat_history' not in session:
            session['chat_history'] = []

        session['chat_history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': time.time()
        })

        # Keep only last 10 exchanges to prevent memory issues
        if len(session['chat_history']) > 10:
            session['chat_history'] = session['chat_history'][-10:]

        session.modified = True

        return jsonify({
            'response': bot_response,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success'})


@app.route('/history', methods=['GET'])
def get_history():
    """Get chat history"""
    return jsonify({
        'history': session.get('chat_history', [])
    })


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch between custom and OpenAI models"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', '').strip().lower()

        if model_type not in ['custom', 'openai']:
            return jsonify({'error': 'Invalid model type. Must be "custom" or "openai"'}), 400

        result = chatbot.switch_model(model_type)

        return jsonify({
            'success': result['success'],
            'message': result['message'],
            'current_model': chatbot.get_current_model_info()
        })

    except Exception as e:
        print(f"Error switching model: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get current model information"""
    try:
        return jsonify({
            'current_model': chatbot.get_current_model_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting GPT-2 Chat Interface...")
    print(f"Device: {chatbot.device}")
    print("Navigate to http://localhost:5000 to use the chat interface")
    app.run(debug=True, host='0.0.0.0', port=5000)
