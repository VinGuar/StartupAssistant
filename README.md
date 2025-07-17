# StartupAssistant

A full-stack AI assistant to help you plan and launch your startup ideas.

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Node.js and npm
- Virtual environment (recommended)

### 1. Backend Setup (FastAPI)

1. Open a terminal in the project root.
2. Create and activate a virtual environment (if not already done):
   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up your environment variables:
   ```sh
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
5. Start the backend server:
   ```sh
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```
   The backend will run at http://localhost:8000

### 2. Frontend Setup (React + Vite)

1. Open a new terminal and go to the frontend directory:
   ```sh
   cd frontend
   ```
2. Install frontend dependencies:
   ```sh
   npm install
   ```
3. Start the frontend dev server:
   ```sh
   npm run dev
   ```
   The frontend will run at http://localhost:3000

### 3. Usage
- Open http://localhost:3000 in your browser.
- Enter your startup idea and get a personalized plan from the AI assistant.

### API Endpoints
- `POST /api/startup` - Submit a startup idea and get AI-generated advice

### Dependencies
The project uses the following key dependencies:
- **FastAPI** - Web framework for the backend
- **LangChain** - AI/LLM integration framework
- **OpenAI** - AI model integration
- **FAISS** - Vector database for similarity search
- **React + Vite** - Frontend framework

### Troubleshooting
- Make sure the backend is running on port 8000 and the frontend on port 3000.
- Ensure your virtual environment is activated when running the backend.
- Check that your `.env` file contains the required API keys.
- If you see import errors, make sure all dependencies are installed: `pip install -r requirements.txt`
- The frontend must call the backend at `/api/startup` (already set up in the code).
- If you see a 404 error, check that the backend is running and the endpoint matches.

### Development
- The backend uses hot reloading with `--reload` flag
- API documentation is available at http://localhost:8000/docs when the server is running
- The project structure follows FastAPI best practices with modular routing

---

Happy building!