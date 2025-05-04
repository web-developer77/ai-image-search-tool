Coding Interview Task: Finding the Best Matches for Highly Specific Image Descriptions
Context
At Vidrush, we're developing AI tools for documentary creation that require finding images matching highly specific descriptions in scripts. These descriptions often include multiple conditions that are unlikely to be fully satisfied in any single existing image (e.g., "Elon Musk standing in a grass field holding an umbrella with a circus in the background").
The challenge is to build a system that can intelligently compromise and find the best possible matches when perfect images don't exist.
Challenge
Build a functional pipeline that takes highly specific image descriptions and returns the closest available matches, ranked by relevance, with clear handling of partial matches.
Requirements

Core Technology
Use LangChain as the orchestration framework for your pipeline
Beyond this requirement, you're free to choose any technologies, APIs, or models that will help solve the problem
Important: Your solution must find and rank existing images only. Do NOT use AI image generation (DALL-E, Midjourney, Stable Diffusion, etc.) to create new images that match the descriptions
API Keys: If your solution requires API keys (for image search, analysis, etc.), please list them in your documentation. We will provide all necessary API keys for the services you choose to use.

Pipeline Functionality
Your solution should:
Accept highly specific image descriptions as input
Implement a strategy to break down complex requirements and search effectively
Return a ranked set of images that best satisfy the description's conditions
Handle the reality that perfect matches likely don't exist

Implementation Goals
Create a working end-to-end pipeline that makes intelligent compromises
Design a modular architecture with clear separation of concerns
Implement a scoring system that weighs different aspects of the description
Write clean, maintainable code with appropriate error handling

Testing Requirements
Your pipeline should find the best possible matches for descriptions like:
"Elon Musk standing in a grass field holding an umbrella with a circus in the background"
"A golden retriever wearing sunglasses riding a skateboard on a beach at sunset"
"An astronaut playing chess with a robot on top of a mountain"
"A Victorian-era woman using a modern smartphone in a library"
Evaluation Criteria
Code quality: Clean, readable, well-structured code
Problem-solving approach: How you handle finding partial matches when perfect matches don't exist
Pipeline design: Effectiveness of your modular architecture
Search strategy: How you break down and prioritize complex requirements
Error handling: How your code deals with failures and edge cases
Deliverables
Complete source code
Setup and configuration instructions, including which API keys are needed
Brief documentation of your approach and architecture

