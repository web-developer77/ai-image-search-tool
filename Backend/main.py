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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")# Convert to int with default value of 10

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

def fetch_descriptions(image_urls, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    descriptions = []
    
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
            descriptions.append(response.choices[0].message.content)
        except Exception as e:
            descriptions.append("")
            
    return descriptions

# Function to score similarity using LangChain
def score_similarity(prompt, descriptions):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(
            texts=descriptions,
            embedding=embeddings,
            metadatas=[{"index": i} for i in range(len(descriptions))]
        )
        # Perform similarity search for the prompt
        similar_docs = vector_store.similarity_search_with_score(prompt, k=len(descriptions))
        
        # Print the similarity scores
        for doc, score in similar_docs:
            similarity = 1 - score  # Convert FAISS distance to similarity
            print(f"Description: {doc.page_content}")
            print(f"Similarity: {similarity:.4f}\n")

        # Find the most similar description
        most_similar_doc = similar_docs[0][0].page_content
        most_similar_score = 1 - similar_docs[0][1]
        print(f"Most similar description: {most_similar_doc} (Similarity: {most_similar_score:.4f})")
        
        return similar_docs
        
        
        # prompt_embedding = embeddings.embed_query(prompt)
        # print("" + prompt_embedding)
        # scores = []
        # valid_descriptions = []
        
        # for desc in descriptions:
        #     if desc:  # Only score non-empty descriptions
        #         desc_embedding = embeddings.embed_query(desc)
        #         score = cosine_similarity([prompt_embedding], [desc_embedding])[0][0]
        #         scores.append(score)
        #         valid_descriptions.append(desc)
        #     else:
        #         scores.append(0.0)
        #         valid_descriptions.append("")
                
        # return valid_descriptions, scores
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return descriptions, [0.0] * len(descriptions)

# def get_unique_urls(image_urls):
#     seen_urls = set()
#     seen_domains = set()
#     unique_urls = []
    
#     for url in image_urls:
#         # Check for exact URL duplicates
#         if url.lower() not in seen_urls:
#             # Extract domain to avoid near-duplicates from same site
#             domain = urlparse(url).netloc
#             if domain not in seen_domains:
#                 try:
#                     # Validate URL accessibility
#                     response = requests.head(url, timeout=5)
#                     if response.status_code == 200 and "image" in response.headers.get("content-type", "").lower():
#                         unique_urls.append(url)
#                         seen_urls.add(url.lower())
#                         seen_domains.add(domain)
#                 except requests.RequestException:
#                     continue
                    
#     return unique_urls

def search_images(prompt: str): 
    try:     
        # Fetch image URLs
        image_urls = fetch_image_urls(prompt, CUSTOM_SEARCH_API_KEY, CX_ID, IMAGE_COUNT)
        
        descriptions = fetch_descriptions(image_urls, OPENAI_API_KEY)
        
        similar_docs = score_similarity(prompt, descriptions)
        
        print(similar_docs)
        
        
                                         
        # Sort image URLs based on similarity scores
        # ranked_pairs = sorted(zip(image_urls, scores), key=lambda x: x[1], reverse=True)
        # sorted_urls = [url[0] for url in ranked_pairs]
        
                    
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

