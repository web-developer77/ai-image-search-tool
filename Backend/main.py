from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path=".env.local")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
CX_ID= os.getenv("CX_ID")
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
IMAGE_COUNT = os.getenv("IMAGE_COUNT")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    prompt: str


def search_images(prompt: str):
    url = "https://www.googleapis.com/customsearch/v1"
     
    params = {
        "q": prompt,
        "searchType": "image",
        "key": CUSTOM_SEARCH_API_KEY,
        "cx": CX_ID,
        "num": IMAGE_COUNT
    }
 
    try:
        response = requests.get(url, params=params)
                      
        data = response.json()
        image_urls = [item['link'] for item in data.get('items', [])]
        descriptions = []
                   
        # Process each image individually
        for i, url in enumerate(image_urls, 1):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url}
                                }
                            ]
                        }
                    ],
                    max_tokens=200
                )
                description = response.choices[0].message.content
                descriptions.append(description)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                descriptions.append("")  # Add empty description for failed images
                
        # Embed the prompt and descriptions
        # prompt_embedding = embeddings.embed_query(prompt)
        # desc_embeddings = [embeddings.embed_query(desc) for desc in descriptions]
        
        # scores = cosine_similarity([prompt_embedding], desc_embeddings)[0]
        
        # Sort image URLs based on similarity scores
        # ranked_pairs = sorted(zip(image_urls, scores), key=lambda x: x[1], reverse=True)
        # sorted_urls = [url[0] for url in ranked_pairs]
        
        print(image_urls)
        print(descriptions)
                    
        return image_urls
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    try:
        images = search_images(request.prompt)
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

