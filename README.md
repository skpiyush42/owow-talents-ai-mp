owow-talents-ai-mp
===============

A Python-based ETL pipeline and Fast API service with multi-db setup
---------------

This repo contains code for the take-home assignment for the role of GCP data engineer at OWOW talents.

## 📂 Project Structure
├── src/   
│  ├── pipeline/&ensp;&ensp;# ETL / orchestration scripts              
│  ├── api/&ensp;&ensp;&ensp;# FastAPI app   
│  ├── utils/                  
│  ├── data/  # contains the milvus lite db and sample file used in pipeline  
│  ├── secrets/&ensp;&ensp;&ensp;# store your SA key json for GCP project  
│  ├── docker-compose.yml&ensp;&ensp;&ensp; Docker compose file             
│  ├── README.md   
│  ├── architecture_diagram.png   
│  ├── architecture.md   
└  └── scaling_plan.md  

## ℹ️ Database details

Code repo contains the following DB setup
1. Raw chat data from users (user_id, message, timestamp) ----> Mongodb
2. 1024-dim vector embeddings ----> Milvus Lite
3. Relationship mappings linking users, campaigns, etc. ----> Neo4j
4. Analytics, aggregated interaction data ----> BigQuery

## ℹ️ ETL-pipeline
1. This code sets up the databases.
2. Loads the data from csv stored on GCS.
3. Creates a relationship in the Neo4j DB
4. Loads vector embeddings into Milvus Lite
5. Uploads CSV data into BigQuery table for analytics

## ℹ️ Fast API service
1. Runs the queries against the data in db's created by the ETL pipeline
2. Has a Fast API service that implements the GET HTTP method to retrieve the query results 
3. The following are the queries 
  ● Retrieve top 5 most similar users (via Milvus vector search).
  ● Fetch campaigns connected to those users (via Neo4j).
  ● Return results ranked by engagement frequency (from analytics DB).

## 📦 Deployment Guide
### Prerequisites before execution on the cloud
#### ● Make sure you have Docker Desktop installed.
#### ● Fill the config as per your GCP project details
#### ● Upload your SA JSON in secrets/ dir
#### ● Make sure the service account you are using has the following permissions: 1. roles/bigquery.jobUser 2. roles/storage.objectUser
#### ● Run "docker compose up" at the root directory.
Notes: 
1. Once Docker Compose has run successfully, you will see the containers in Docker Desktop.
2. No need to run docker compose up every time you make changes to the code. Just restart the relevant service container.

