# Recipe_Assistance
## Table of content
- [Project goal](#project-goal)
- [Data source](#data-source)
- [Design Choices](#design-choices)
- [Enviroment Setup](#enviroment-setup)
- [Contact](#contact)

##  Problem statement
In today's digital age, food enthusiasts and home cooks are increasingly turning to online platforms for culinary inspiration, recipe discovery, and cooking tips. However, current solutions often lack the ability to provide personalized, contextually accurate responses that cater to specific user queries. Users face challenges in navigating large recipe datasets to find the exact information they need, such as adapting recipes to dietary restrictions, suggesting ingredient substitutions, or providing step-by-step cooking guidance.

This project addresses the need for a more interactive, intelligent system capable of answering culinary-related questions with precision and relevance. By leveraging a Retrieval-Augmented Generation (RAG) system, the goal is to bridge the gap between users' questions and comprehensive recipe data. The solution will integrate advanced natural language processing (NLP) techniques and efficient retrieval mechanisms, offering users an engaging and informative experience that enhances their culinary journey.

The key challenge lies in building a system that not only retrieves relevant data from a vast recipe dataset but also generates meaningful, context-aware answers, tailored to each user’s unique query.
## Project goal
This project aims to develop a Retrieval-Augmented Generation (RAG) system that answers culinary-related questions by leveraging a comprehensive recipe dataset. The system will combine the power of natural language processing (NLP) and retrieval mechanisms based on Langchain to provide accurate, contextually relevant, and detailed answers. By efficiently retrieving relevant recipe information and generating responses, the system aims to assist users in discovering new culinary insights, enhancing their cooking experiences, and deepening their engagement with the platform.
For more details, you can access my notebook

## Data source
I use the Recipe dataset from Kaggle.
The project utilizes a recipe dataset sourced from Kaggle, specifically the recipes.csv file, which contains information on over 1,000 recipes. The dataset includes fields such as recipe names, preparation times, cooking times, ratings, ingredients, and step-by-step instructions. For the purpose of this project, I pre-processed and used 100 recipes. The relevant fields are:
- recipe_name: The name of the recipe. (String)
- prep_time: The amount of time required to prepare the recipe. (Integer)
- cook_time: The amount of time required to cook the recipe. (Integer)
- total_time: The total amount of time required to prepare and cook the recipe. (Integer)
- servings: The number of servings the recipe yields. (Integer)
-  ingredients: A list of ingredients required to make the recipe. (List)
- directions: A list of directions for preparing and cooking the recipe. (List)
- rating: The recipe rating. (Float)
- url: The recipe URL. (String)
- cuisine_path: The recipe cuisine path. (String)
- nutrition: The recipe nutrition information. (Dictionary)
- timing: The recipe timing information. (Dictionary)
- img_src: Links to the image of the recipe

## Design choices
The project implementation was carried out in several phases:
- Environment Setup
- Data Processing layer
- Chucking and Embedding layer 
- Driver layer
- Conversation layer

1. Environment Setup:
- Weaviate Docker: Used Weaviate as the knowledge base to store the recipe data.
- Redis Docker: Used Redis as an in-memory database to store user chat histories.
- Ollama Docker: Integrated Ollama for local LLM model access when required.

2. Data Processing layer:
- Document processing: Merged specific columns from the recipe dataset to feed into the knowledge database.
- EDA: Examined document text lengths to determine optimal chunk sizes for processing.

3. Chungking and Embeding layer:
- Chunking Strategy: Experimented with different chunk sizes and overlap strategies to ensure no important information is lost during retrieval. Two chunking methods from Langchain were explored: RecursiveCharacterTextSplitter and CharacterTextSplitter.
- Embedding Model Choice: Tested embeddings from OpenAI’s models and the Ollama library.

4. Driver layer
- Driver library: Developed redisdb.py and weaviatedb.py for seamless interaction with the respective databases.

5. Conversation Layer
- Config: Created a config/prompt_config.yml file to facilitate prompt customization.
- Tracing: Used Langsmith to trace chatbot interactions, especially for debugging prompts. You can visit [this link](https://smith.langchain.com/public/587de497-ddb0-456c-a3e3-f4f65519fb86/r) to see how Langsmith helps to visualize the chains.
- Chatbot chains: Designed chatbot chains using Langchain to retrieve relevant documents, generate answers, and manage conversation histories.

## Enviroment Setup
1. Create conda environment and install necessary libraries
```Bash
conda create -n recipe python=3.10
conda activate  recipe
pip install -r requirements.txt
```
2. Install vector embedding DB (weaviate)
```Bash
cd db/weaviate
docker compose up -d
```
3. Install redis
```Bash
docker pull redis/redis-stack-server
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```
4. Create .env
- OPENAI_API_KEY= "you api key"
- LANGCHAIN_API_KEY = "you api key"
5. Install Ollama
Based on you device, you can choose the suitable methods to install Ollama by visit [this blog](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)
```Bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
## Contact
Created by [Huynh Viet Cuong](https://cuonghv0298.github.io/) - feel free to contact me by email vietcuong.ip@gmail.com!



