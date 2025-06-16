# Ecommerce Product Recommendation Application using Hugging Face LLM

## Project Overview

This project aims to develop an intelligent e-commerce product recommendation system that suggests products based on users' preferences. It leverages a Large Language Model (LLM) integrated through the LangChain framework and a FAISS vector database for efficient semantic search over product data.

## Application Demo

The application provides two distinct methods for users to receive product recommendations:

1.  **Manual Input:** Users can input their product preferences (e.g., department, category, maximum price) into provided fields to receive a filtered list of products.
    ![Manual Recommendation Demo](streamlit-app-2024-02-05-20-02-45.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-45.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

2.  **Chatbot Interaction:** Users can engage in a natural language conversation with an AI assistant. The chatbot intelligently interprets user queries and preferences to provide relevant product recommendations.
    ![Chatbot Demo](streamlit-app-2024-02-05-20-02-31.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-31.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

## Project Structure and File Explanation

The repository is organized into `backend` and `frontend` directories, along with other essential configuration files:

 .
├── backend/
│   ├── app.py
│   ├── bq-results-20240205-004748-1707094090486.csv
│   ├── chatbot.py
│   ├── dockerfile
│   ├── requirements.txt
│   └── models/
│       └── all-MiniLM-L6-v2/ (Contains locally downloaded embedding model files)
├── frontend/
│   ├── app.py
│   ├── dockerfile
│   ├── requirements.txt
├── .gitignore
└── README.md



* **`backend/app.py`**:
    This file contains the backend code, responsible for server-side logic, API endpoints, and integrating the recommendation system with potential external calls.
* **`backend/bq-results-20240205-004748-1707094090486.csv`**:
    This is the primary dataset used in the project. It was obtained from Google Cloud Platform's BigQuery database, specifically from the `thelook_ecommerce` table, combining data from `order_items`, `inventory_items`, and `users`.
* **`backend/chatbot.py`**:
    This core file implements the recommendation system using the LangChain framework. It initializes the Hugging Face `google/flan-t5-large` LLM and the `sentence-transformers/all-MiniLM-L6-v2` embedding model, sets up the FAISS vector store with your product data, and defines both conversational and structured recommendation chains.
* **`backend/dockerfile`**:
    This Dockerfile is used to build a Docker image for the backend application, ensuring a consistent and isolated environment for its execution. It includes instructions on how to set up the environment and dependencies needed.
* **`backend/requirements.txt`**:
    This file lists all the Python dependencies required for the backend application. These dependencies can be easily installed using a package manager like `pip`.
* **`backend/models/all-MiniLM-L6-v2/`**:
    This directory is crucial for the project's operation. It's where the manually downloaded files for the `sentence-transformers/all-MiniLM-L6-v2` embedding model should be placed. This step helps in avoiding `HTTP 429` (Too Many Requests) errors and ensures stable model loading.
* **`frontend/app.py`**:
    This is the main script for the frontend of the application, developed using the Streamlit framework. It creates the user interface, including sections for manual product input and the chat interface, and integrates with the backend functionality.
* **`frontend/dockerfile`**:
    Similar to the backend Dockerfile, this file is used to build a Docker image for the frontend application, providing a containerized environment for the Streamlit app.
* **`frontend/requirements.txt`**:
    This file lists the Python dependencies required for the frontend application. These dependencies can be installed using `pip`.
* **`.gitignore`**:
    This standard file specifies intentionally untracked files and directories that Git should ignore (e.g., virtual environments, cache files, `.env` files containing sensitive information).
* **`README.md`**:
    This Markdown file provides comprehensive documentation for the project, including setup instructions, how to use the application, dependencies, and any other relevant details.

## Setup and How to Run the Application

Follow these steps to set up and run the e-commerce product recommendation application on your local machine.

### Prerequisites

* Python 3.9 or higher
* `pip` (Python package installer)
* Git
* Docker (Optional, if you prefer containerized deployment)

### Step 1: Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/MdShadil07/ProductRecommendation.git](https://github.com/MdShadil07/ProductRecommendation.git)
cd ProductRecommendation



Okay, here is the complete and updated README.md file. You can directly copy and paste this content into your README.md file in the project root.

Markdown

# Ecommerce Product Recommendation Application using Hugging Face LLM

## Project Overview

This project aims to develop an intelligent e-commerce product recommendation system that suggests products based on users' preferences. It leverages a Large Language Model (LLM) integrated through the LangChain framework and a FAISS vector database for efficient semantic search over product data.

## Application Demo

The application provides two distinct methods for users to receive product recommendations:

1.  **Manual Input:** Users can input their product preferences (e.g., department, category, maximum price) into provided fields to receive a filtered list of products.
    ![Manual Recommendation Demo](streamlit-app-2024-02-05-20-02-45.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-45.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

2.  **Chatbot Interaction:** Users can engage in a natural language conversation with an AI assistant. The chatbot intelligently interprets user queries and preferences to provide relevant product recommendations.
    ![Chatbot Demo](streamlit-app-2024-02-05-20-02-31.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-31.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

## Project Structure and File Explanation

The repository is organized into `backend` and `frontend` directories, along with other essential configuration files:

.
├── backend/
│   ├── app.py
│   ├── bq-results-20240205-004748-1707094090486.csv
│   ├── chatbot.py
│   ├── dockerfile
│   ├── requirements.txt
│   └── models/
│       └── all-MiniLM-L6-v2/ (Contains locally downloaded embedding model files)
├── frontend/
│   ├── app.py
│   ├── dockerfile
│   ├── requirements.txt
├── .gitignore
└── README.md


* **`backend/app.py`**:
    This file contains the backend code, responsible for server-side logic, API endpoints, and integrating the recommendation system with potential external calls.
* **`backend/bq-results-20240205-004748-1707094090486.csv`**:
    This is the primary dataset used in the project. It was obtained from Google Cloud Platform's BigQuery database, specifically from the `thelook_ecommerce` table, combining data from `order_items`, `inventory_items`, and `users`.
* **`backend/chatbot.py`**:
    This core file implements the recommendation system using the LangChain framework. It initializes the Hugging Face `google/flan-t5-large` LLM and the `sentence-transformers/all-MiniLM-L6-v2` embedding model, sets up the FAISS vector store with your product data, and defines both conversational and structured recommendation chains.
* **`backend/dockerfile`**:
    This Dockerfile is used to build a Docker image for the backend application, ensuring a consistent and isolated environment for its execution. It includes instructions on how to set up the environment and dependencies needed.
* **`backend/requirements.txt`**:
    This file lists all the Python dependencies required for the backend application. These dependencies can be easily installed using a package manager like `pip`.
* **`backend/models/all-MiniLM-L6-v2/`**:
    This directory is crucial for the project's operation. It's where the manually downloaded files for the `sentence-transformers/all-MiniLM-L6-v2` embedding model should be placed. This step helps in avoiding `HTTP 429` (Too Many Requests) errors and ensures stable model loading.
* **`frontend/app.py`**:
    This is the main script for the frontend of the application, developed using the Streamlit framework. It creates the user interface, including sections for manual product input and the chat interface, and integrates with the backend functionality.
* **`frontend/dockerfile`**:
    Similar to the backend Dockerfile, this file is used to build a Docker image for the frontend application, providing a containerized environment for the Streamlit app.
* **`frontend/requirements.txt`**:
    This file lists the Python dependencies required for the frontend application. These dependencies can be installed using `pip`.
* **`.gitignore`**:
    This standard file specifies intentionally untracked files and directories that Git should ignore (e.g., virtual environments, cache files, `.env` files containing sensitive information).
* **`README.md`**:
    This Markdown file provides comprehensive documentation for the project, including setup instructions, how to use the application, dependencies, and any other relevant details.

## Setup and How to Run the Application

Follow these steps to set up and run the e-commerce product recommendation application on your local machine.

### Prerequisites

* Python 3.9 or higher
* `pip` (Python package installer)
* Git
* Docker (Optional, if you prefer containerized deployment)

### Step 1: Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/MdShadil07/ProductRecommendation.git](https://github.com/MdShadil07/ProductRecommendation.git)
cd ProductRecommendation
Step 2: Set up Environment Variables
Create a file named .env in the backend/ directory. This file will store your Hugging Face API token securely.


# backend/.env
HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
Replace "YOUR_HUGGINGFACE_API_TOKEN" with your actual Hugging Face API token. You can generate one from your Hugging Face settings page.

Step 3: Download the Embedding Model Locally (CRITICAL for avoiding errors!)
To prevent HTTP 429 (Too Many Requests) errors and ensure consistent model loading, you must manually download the embedding model files:

Create the model directory: From your project root directory (ProductRecommendation), navigate to the backend folder and create the necessary directory structure:
Bash

mkdir -p backend/models/all-MiniLM-L6-v2


Download all model files: Go to the Hugging Face model repository for sentence-transformers/all-MiniLM-L6-v2 in your web browser: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main Click on the "Files & Versions" tab. You will see a list of files (e.g., config.json, pytorch_model.bin, tokenizer.json, vocab.txt, etc.). Download EVERY SINGLE FILE listed there.
Place downloaded files: Move all the files you just downloaded directly into the backend/models/all-MiniLM-L6-v2/ directory. Ensure no subfolders are created inside all-MiniLM-L6-v2; just place the files themselves.


Okay, here is the complete and updated README.md file. You can directly copy and paste this content into your README.md file in the project root.

Markdown

# Ecommerce Product Recommendation Application using Hugging Face LLM

## Project Overview

This project aims to develop an intelligent e-commerce product recommendation system that suggests products based on users' preferences. It leverages a Large Language Model (LLM) integrated through the LangChain framework and a FAISS vector database for efficient semantic search over product data.

## Application Demo

The application provides two distinct methods for users to receive product recommendations:

1.  **Manual Input:** Users can input their product preferences (e.g., department, category, maximum price) into provided fields to receive a filtered list of products.
    ![Manual Recommendation Demo](streamlit-app-2024-02-05-20-02-45.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-45.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

2.  **Chatbot Interaction:** Users can engage in a natural language conversation with an AI assistant. The chatbot intelligently interprets user queries and preferences to provide relevant product recommendations.
    ![Chatbot Demo](streamlit-app-2024-02-05-20-02-31.mp4)
    *(Replace `streamlit-app-2024-02-05-20-02-31.mp4` with a link to a GIF or video demonstrating this feature, if hosted publicly.)*

## Project Structure and File Explanation

The repository is organized into `backend` and `frontend` directories, along with other essential configuration files:

.
├── backend/
│   ├── app.py
│   ├── bq-results-20240205-004748-1707094090486.csv
│   ├── chatbot.py
│   ├── dockerfile
│   ├── requirements.txt
│   └── models/
│       └── all-MiniLM-L6-v2/ (Contains locally downloaded embedding model files)
├── frontend/
│   ├── app.py
│   ├── dockerfile
│   ├── requirements.txt
├── .gitignore
└── README.md


* **`backend/app.py`**:
    This file contains the backend code, responsible for server-side logic, API endpoints, and integrating the recommendation system with potential external calls.
* **`backend/bq-results-20240205-004748-1707094090486.csv`**:
    This is the primary dataset used in the project. It was obtained from Google Cloud Platform's BigQuery database, specifically from the `thelook_ecommerce` table, combining data from `order_items`, `inventory_items`, and `users`.
* **`backend/chatbot.py`**:
    This core file implements the recommendation system using the LangChain framework. It initializes the Hugging Face `google/flan-t5-large` LLM and the `sentence-transformers/all-MiniLM-L6-v2` embedding model, sets up the FAISS vector store with your product data, and defines both conversational and structured recommendation chains.
* **`backend/dockerfile`**:
    This Dockerfile is used to build a Docker image for the backend application, ensuring a consistent and isolated environment for its execution. It includes instructions on how to set up the environment and dependencies needed.
* **`backend/requirements.txt`**:
    This file lists all the Python dependencies required for the backend application. These dependencies can be easily installed using a package manager like `pip`.
* **`backend/models/all-MiniLM-L6-v2/`**:
    This directory is crucial for the project's operation. It's where the manually downloaded files for the `sentence-transformers/all-MiniLM-L6-v2` embedding model should be placed. This step helps in avoiding `HTTP 429` (Too Many Requests) errors and ensures stable model loading.
* **`frontend/app.py`**:
    This is the main script for the frontend of the application, developed using the Streamlit framework. It creates the user interface, including sections for manual product input and the chat interface, and integrates with the backend functionality.
* **`frontend/dockerfile`**:
    Similar to the backend Dockerfile, this file is used to build a Docker image for the frontend application, providing a containerized environment for the Streamlit app.
* **`frontend/requirements.txt`**:
    This file lists the Python dependencies required for the frontend application. These dependencies can be installed using `pip`.
* **`.gitignore`**:
    This standard file specifies intentionally untracked files and directories that Git should ignore (e.g., virtual environments, cache files, `.env` files containing sensitive information).
* **`README.md`**:
    This Markdown file provides comprehensive documentation for the project, including setup instructions, how to use the application, dependencies, and any other relevant details.

## Setup and How to Run the Application

Follow these steps to set up and run the e-commerce product recommendation application on your local machine.

### Prerequisites

* Python 3.9 or higher
* `pip` (Python package installer)
* Git
* Docker (Optional, if you prefer containerized deployment)

### Step 1: Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/MdShadil07/ProductRecommendation.git](https://github.com/MdShadil07/ProductRecommendation.git)
cd ProductRecommendation
Step 2: Set up Environment Variables
Create a file named .env in the backend/ directory. This file will store your Hugging Face API token securely.

# backend/.env
HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
Replace "YOUR_HUGGINGFACE_API_TOKEN" with your actual Hugging Face API token. You can generate one from your Hugging Face settings page.

Step 3: Download the Embedding Model Locally (CRITICAL for avoiding errors!)
To prevent HTTP 429 (Too Many Requests) errors and ensure consistent model loading, you must manually download the embedding model files:

Create the model directory: From your project root directory (ProductRecommendation), navigate to the backend folder and create the necessary directory structure:
Bash

mkdir -p backend/models/all-MiniLM-L6-v2
Download all model files: Go to the Hugging Face model repository for sentence-transformers/all-MiniLM-L6-v2 in your web browser: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main Click on the "Files & Versions" tab. You will see a list of files (e.g., config.json, pytorch_model.bin, tokenizer.json, vocab.txt, etc.). Download EVERY SINGLE FILE listed there.
Place downloaded files: Move all the files you just downloaded directly into the backend/models/all-MiniLM-L6-v2/ directory. Ensure no subfolders are created inside all-MiniLM-L6-v2; just place the files themselves.
Step 4: Install Dependencies
Navigate into the backend and frontend directories one by one and install their respective Python dependencies using pip:

Bash

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd .. # Go back to the project root directory

# Install frontend dependencies
cd frontend
pip install -r requirements.txt
cd .. # Go back to the project root directory
Step 5: Run the Application
You can run the application either directly using Python/Streamlit or using Docker. For development, running directly is usually simpler.

Option A: Running Directly (Recommended for Development)
Since your frontend/app.py (Streamlit) directly imports the chatbot_manager from backend/chatbot.py, you primarily need to run the Streamlit application. The backend logic will be initialized automatically when the frontend starts.

Navigate to the frontend directory:
Bash

cd frontend
Run the Streamlit application:
Bash

streamlit run app.py
This command will start the Streamlit server and open the application in your default web browser (usually at http://localhost:8501).
Option B: Running with Docker (For Containerized Deployment)
If your backend/app.py is intended to be a separate API service (e.g., Flask or FastAPI) that the frontend then calls, you would typically use docker-compose. If your current setup has the frontend app.py directly importing chatbot.py, you might only need to containerize the frontend.

Assuming a Microservice Architecture (Backend API + Frontend App):

Create docker-compose.yml (in your project root ProductRecommendation/):

YAML

# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    container_name: product_recommender_backend
    ports:
      - "5000:5000" # Example port for a backend API (adjust as per your backend/app.py)
    env_file:
      - ./backend/.env
    volumes:
      - ./backend/models:/app/models # Mount local model directory into container
    command: python app.py # Or 'flask run' or 'uvicorn app:app --host 0.0.0.0 --port 5000' - Adjust this based on how your backend app.py is served.

  frontend:
    build: ./frontend
    container_name: product_recommender_frontend
    ports:
      - "8501:8501" # Default Streamlit port
    environment:
      - BACKEND_URL=http://backend:5000 # This assumes backend is a separate service
    depends_on:
      - backend # Ensures backend service starts before frontend
    command: streamlit run app.py --server.port 8501 --server.enableCORS false
(Note: Adjust the command for the backend service in docker-compose.yml based on how your backend/app.py is designed to be run as an API server. If backend/app.py simply initializes Python objects without serving an HTTP API, this docker-compose setup might need adjustment to put all logic into the frontend container.)

Build and Run with Docker Compose:
From the root of your project directory (ProductRecommendation):

Bash

docker-compose up --build
This command will build the Docker images (if not already built) and start both the backend and frontend containers.

Access the Application:
Once the containers are running, open your web browser and navigate to http://localhost:8501 to access the Streamlit frontend.

Contact
GitHub Profile: MdShadil07
Email: shadilrock77@gmail.com








