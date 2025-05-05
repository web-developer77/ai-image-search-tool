1. Please run below command in Frontend directory to run frontend as development mode
- Make .env.local file in root of Frontend directory and set value like below

BACKEND_API_URL=http://localhost:8000

- Install node_modules
`npm install`
- Run frontend as dev mode
`npm run dev`

2. Please run below command in Backend directory to run backend 
- Make .env.local file in root of Backend directory and set value like below
PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}
CUSTOM_SEARCH_API_KEY=${CUSTOM_SEARCH_API_KEY}
CX_ID=${CX_ID}
IMAGE_COUNT=10

- Install virtual environment
`python -m venv venv`
- Activate venv in windows
`.\venv\Scripts\activate`
- Install depencencies
`pip install -r requirements.txt`
- Run backend
`python main.py`