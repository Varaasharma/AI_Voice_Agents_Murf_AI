# AI Voice Agent

A real-time conversational AI voice agent built with FastAPI and modern web technologies. This project demonstrates a complete voice conversation pipeline with speech-to-text, AI response generation, and text-to-speech synthesis.

## Features

- **One-Button Conversation Loop**: Simple tap-to-record interface that automatically restarts after each response.
- **Real-Time Speech Processing**: Live audio recording, transcription, and playback.
- **Contextual Conversations**: Maintains conversation history for context-aware responses.
- **Robust Error Handling**: Automatic fallbacks for API failures using browser TTS.
- **Modern UI**: Responsive, accessible interface with smooth animations and visual feedback.
- **Session Management**: Unique session IDs for conversation continuity.

## Architecture

### Backend (FastAPI)
- **Speech-to-Text**: AssemblyAI API for audio transcription.
- **AI Response Generation**: Google Gemini 1.5 Flash for contextual replies.
- **Text-to-Speech**: Murf AI for high-quality voice synthesis.
- **Session Management**: In-memory chat history storage.
- **Error Handling**: Graceful degradation with fallback mechanisms.

### Frontend (Vanilla JavaScript)
- **Audio Recording**: Web Audio API with MediaRecorder.
- **Real-Time UI**: Dynamic state management with visual feedback.
- **Audio Playback**: Automatic audio streaming and fallback TTS.
- **Responsive Design**: Modern CSS with animations and accessibility features.

## Project Structure

```
voice-agents/
├── backend/
│   └── main.py              # FastAPI server with voice processing endpoints
├── frontend/
│   ├── index.html           # Main application interface
│   ├── static/
│   │   └── script.js        # Frontend logic and audio handling
│   └── README.md            # This file
├── uploads/                 # Temporary audio file storage
├── venv/                    # Python virtual environment
├── .gitignore              # Git ignore patterns
└── .env                     # Environment variables (not tracked)
```

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- Microphone access in your browser

### 1. Clone the Repository
```bash
git clone https://github.com/Varaasharma/AI_Voice_Agents_Murf_AI.git
cd voice-agents
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-multipart assemblyai murf google-generativeai python-dotenv
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
MURF_API_KEY=your_murf_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
```bash
# Start the FastAPI server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application
Open your browser and navigate to `http://localhost:8000`

## API Keys Required

- **AssemblyAI**: For speech-to-text transcription
- **Murf AI**: For high-quality text-to-speech synthesis
- **Google Gemini**: For AI-powered conversation responses

## How It Works

### Conversation Flow
1. **Recording**: User taps the microphone button to start recording
2. **Transcription**: Audio is sent to AssemblyAI for speech-to-text conversion
3. **Context Building**: Conversation history is retrieved and formatted for context
4. **AI Response**: Gemini generates a contextual reply based on the conversation
5. **Voice Synthesis**: Murf converts the text response to natural speech
6. **Playback**: Audio is streamed back to the browser and automatically played
7. **Loop Restart**: Recording automatically resumes for the next turn

### Error Handling
- **STT Failure**: Falls back to browser TTS with error message
- **LLM Failure**: Returns fallback response with user's transcribed text
- **TTS Failure**: Uses browser TTS for response playback
- **Server Errors**: Graceful degradation with audible error messages

## UI States

The interface provides visual feedback for different states:
- **Idle**: Blue microphone button, ready to record
- **Recording**: Orange button with pulsing animation and audio bars
- **Processing**: Blue button with processing animation
- **Playing**: Green button while audio is playing
- **Error**: Red indicator for error states

## Security Features


- Session-based conversation isolation
- Environment variable protection for API keys
- Input validation and sanitization

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints
- `GET /`: Serves the main application interface
- `GET /health`: Health check with API status
- `POST /agent/chat/{session_id}`: Main conversation endpoint

### Testing
The application includes a health endpoint to verify API connectivity:
```bash
curl http://localhost:8000/health
```

## Browser Compatibility

- Modern browsers with Web Audio API support
- Chrome, Firefox, Safari, Edge (latest versions)
- Requires microphone permissions
- HTTPS recommended for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of a learning series. Please check the repository for specific licensing information.

## Troubleshooting

### Common Issues
- **Microphone Access**: Ensure browser permissions are granted
- **API Errors**: Check your `.env` file and API key validity
- **Audio Playback**: Verify browser audio settings and permissions
- **CORS Issues**: Check server configuration for production deployment

### Debug Mode
Enable detailed logging by checking the browser console and server logs for error details.

---



