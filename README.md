# Overview

An autonomous AI analyst capable of answering complex business questions by navigating both unstructured documents (PDFs/Reports) and structured relational databases (SQL) while keeping track of every request cost throughout the entire process.

## The Problem

You are an AI Data Engineer at a financial tech startup. Non-technical executives need to ask questions like:

> *"What was Apple's total hardware revenue last year, and what did their Q3 report say about supply chain risks?"*
> 

It requires an agent that can dynamically write SQL to calculate the revenue, and simultaneously query a Vector Database to read the Q3 report. A standard LLM cannot answer this hence the need for the RAG Data Analyst

## Prerequisites

- Docker and Docker Compose Installed
- Docker Desktop (for local Qdrant)
- Python 3.10+ (for local dependency management and any script inspection)
- A `.env` file created from `.env.example`
- Your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## To Run

1. Clone this repository
2. Create and activate a virtual environment:
    ````bash
    python -m venv venv
    ````
	````bash
    source venv/bin/activate
    ````
3. Install dependencies:
    ````bash
    pip install -r requirements.txt
    ````
4. Create a `.env` at the project root from `.env.example` and add your `GOOGLE_API_KEY`


5. Run Qdrant (local development)
 - Start Qdrant via docker-compose (provided):

	````bash
    docker compose up --build
    ````

 - If you prefer a managed Qdrant, set `QDRANT_URL` to the HTTP endpoint.

7. Prepare documents for the vector embedding

- Run the script to extract texts from PDFs and clean texts:

    ````bash
    python etl/extract_pdfs.py
    ````
    ````bash
    python etl/clean.py
    ````

8. Prepare vectors & indexes
 - Run the ETL pipeline to create the collection, payload indexes, and upload vectors:

	````bash
    python etl/pipeline.py
    ````

	The pipeline creates the collection and ensures `company_name` and `document_year` payload indexes.

9. Create SQLite DB

    ````bash
    python etl/extract_financials.py
    ````


10. Run the API locally
 - Start the FastAPI app with Uvicorn:

	source venv/bin/activate
	uvicorn app.main:app --host 0.0.0.0 --port 8080

 - Open http://localhost:8080/docs to try the `/query` endpoint.