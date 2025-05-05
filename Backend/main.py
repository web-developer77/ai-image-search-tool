from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv(dotenv_path=".env.local")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
CX_ID= os.getenv("CX_ID")
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
IMAGE_COUNT = int(os.getenv("IMAGE_COUNT", "10"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    
    # Function to fetch image URLs from Google Custom Search API
def fetch_image_urls(query, api_key, cx_id, num_results=10):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "searchType": "image",
        "key": api_key,
        "cx": cx_id,
        "num": min(num_results, 10),  # Max 10 results per request
        "filter": "1",  # Enable duplicate content filter
        "safe": "off",
        "imgSize": "large"
    }
    
    image_urls = []
    start = 1
    while len(image_urls) < num_results:
        params["start"] = start
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "items" not in data:
                break
                
            for item in data["items"]:
                image_urls.append(item["link"])
                
            start += 10
            if start > 100:  # API limit: max 100 results
                break
                
        except Exception as e:
            print(f"Error fetching images: {e}")
            break
            
    return image_urls[:num_results]

def fetch_image_data(image_urls, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    image_data = []
            
    for url in image_urls:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the content of this image in detail."},
                            {"type": "image_url", "image_url": {"url": url}}
                        ]
                    }
                ],
                max_tokens=200
            )
            image_data.append({ "url": url, "description": response.choices[0].message.content })
        except Exception as e:
            image_data.append({ "url": url, "description": "" })
            
    return image_data

# Function to score similarity using LangChain
def fetch_sorted_image_urls(prompt, image_data):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        sorted_image_urls = []
        # Extract descriptions
        descriptions = [item["description"] for item in image_data]

        # Generate embeddings
        prompt_embedding = embeddings.embed_query(prompt)
        description_embeddings = embeddings.embed_documents(descriptions)

        # Calculate cosine similarity
        similarities = cosine_similarity([prompt_embedding], description_embeddings)[0]

        # Create results list
        results = [
            {"url": item["url"], "description": item["description"], "similarity": sim}
            for item, sim in zip(image_data, similarities)
        ]

        # Sort results by similarity (descending)
        sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        for item in sorted_results:
            sorted_image_urls.append(item["url"])
        
        return sorted_image_urls     
        
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return image_data

def search_images(prompt: str): 
    try:     
        # Fetch image URLs
        image_urls = fetch_image_urls(prompt, CUSTOM_SEARCH_API_KEY, CX_ID, IMAGE_COUNT)
        
        image_data = fetch_image_data(image_urls, OPENAI_API_KEY)
        
        sorted_image_urls = fetch_sorted_image_urls(prompt, image_data)
                              
        return sorted_image_urls
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

