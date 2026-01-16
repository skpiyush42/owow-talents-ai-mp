import pandas as pd
from pymongo import MongoClient
from datetime import datetime, UTC
from pathlib import Path
import logging
import sys
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from pymilvus import FieldSchema, CollectionSchema, DataType
from neo4j import GraphDatabase
import os
import time
from pymilvus.exceptions import MilvusException
import numpy as np

# User defined class imports
from config.config_api import Confing
from gcs.gcs_api import GcsOperations
from utils.utils_api import Utils
from bigquery_module.bigquery_api import BigqueryOperations
from google.cloud.exceptions import GoogleCloudError

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ai-marketing-persnlz")
logger.setLevel(logging.INFO)

# Creating required objects from user defined classes
util_func = Utils()

# Read config from YAML file
config_vals = Confing.load_yaml_config(Path("config.yaml"))

# Getting configuration valuation
INPUT_BUCKET_NAME = config_vals.INPUT_BUCKET
FILE_PREFIX = config_vals.FILE_PREFIX
PROJECT_ID = config_vals.PROJECT_ID
DATASET_ID = config_vals.DATASET_ID
STG_TABLE_ID = config_vals.STG_TABLE_ID
TABLE_ID = config_vals.TABLE_ID
FILE_NAME = config_vals.FILE_NAME

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = config_vals.NEO4J_USER
NEO4J_PASSWORD = config_vals.NEO4J_PASSWORD

gcs_ops_input = GcsOperations(INPUT_BUCKET_NAME)
bq_ops = BigqueryOperations(PROJECT_ID)

# defining zero vectore for none values
EMBED_DIM = 1024
ZERO_VECTOR = np.zeros(EMBED_DIM, dtype=np.float32)
            
# Load a pretrained Sentence Transformer model
model_name = 'BAAI/bge-large-en-v1.5'
model = SentenceTransformer(model_name)

def create_mongo_db_collection(db_name, collection_name):
    # Load to Mondgo DB

    # MongoDB Connection
    client = MongoClient(os.getenv("MONGODB_URI"))
    # Creating collection
    collection = client[f"{db_name}"][f"{collection_name}"]

    return collection


def wait_for_milvus(max_retries=30, delay=2):
    uri = f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}"

    for attempt in range(1, max_retries + 1):
        try:
            client = MilvusClient(uri=uri)
            client.list_collections()
            logger.info("Milvus is ready")
            return client
        except MilvusException:
            logger.info(f"Waiting for Milvus ({attempt}/{max_retries})...")
            time.sleep(delay)

    raise RuntimeError("Milvus did not become ready in time")

def milvus_db_ops():

    # defining DIM for vectore embedding

    DIM = 1024
    collection_name = "user_data"

    client = MilvusClient("/data/milvus_lite.db")

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Define fields

    fields = [
    FieldSchema(
        name="record_id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=512
    ),
    FieldSchema(
        name="user_id",
        dtype=DataType.VARCHAR,
        max_length=512
    ),
    FieldSchema(
        name="text_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=DIM
    )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Text embeddings using Milvus Lite"
    )

    client.create_collection(
    collection_name=collection_name,
    schema=schema
    )

    return client, collection_name

def get_most_similar_users(client, COLLECTION, TARGET_USER_ID):

    # Load collection for querying
    client.load_collection(COLLECTION)

    # Fetch Embeddings for the Given User
    records = client.query(
        collection_name=COLLECTION,
        filter=f'user_id == "{TARGET_USER_ID}"',
        output_fields=["text_embedding"]
    )

    if not records:
        raise ValueError(f"No embeddings found for user_id={TARGET_USER_ID}")

    # Create a Single User Embedding using mean pooling
    user_vectors = np.array(
        [r["text_embedding"] for r in records],
        dtype=np.float32
        )

    query_vector = user_vectors.mean(axis=0)

    # Perform Vector Search (Exclude Same User)

    search_results = client.search(
    collection_name=COLLECTION,
    data=[query_vector.tolist()],
    anns_field="text_embedding",
    limit=5,
    filter=f'user_id != "{TARGET_USER_ID}"',
    output_fields=["user_id"],
    search_params={
        "metric_type": "COSINE"
        }
    )

    similar_users = []

    for hit in search_results[0]:
        similar_users.append({
            "user_id": hit["entity"]["user_id"],
            "score": hit["distance"]
        })

    seen = set()
    unique_users = []

    for u in similar_users:
        if u["user_id"] not in seen:
            seen.add(u["user_id"])
            unique_users.append(u)

    unique_users = unique_users[:5]

    return unique_users

def create_node():
    """Connects to Neo4j and creates a new graph."""
    # Use the driver as a context manager to ensure the connection is closed
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:

        
        # Verify connectivity
        driver.verify_connectivity()
        logger.info("Connection established.")

        # Execute the query to create graph
        # Use database_="system" for administrative commands
        # Use IF NOT EXISTS to prevent errors if the database already exists

        
        
        try:
            # execute to create graph (default database neo4j since we are not using enterprise edition)

            constraints = [
            """
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT campaign_unique IF NOT EXISTS
            FOR (c:Campaign) REQUIRE c.campaign IS UNIQUE
            """,
            """
            CREATE CONSTRAINT event_id_unique IF NOT EXISTS
            FOR (e:Event) REQUIRE e.record_id IS UNIQUE
            """
            ]
            with driver.session() as session:
                for query in constraints:
                    session.run(query)
            
            logger.info("Nodes created successfully.")

        except Exception as e:
            logger.exception(f"An error occurred: {e}")

def create_basic_relationships(df):
    """Connects to Neo4j and creates a new relationship."""

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:

        create_relationship_query = """
                MERGE (u:User {user_id: $user_id})

                MERGE (c:Campaign {campaign: $campaign})

                MERGE (e:Event {record_id: $record_id})
                SET
                e.message = $message,
                e.timestamp = datetime($timestamp)

                MERGE (u)-[:PERFORMED]->(e)
                MERGE (e)-[:PART_OF]->(c)
                MERGE (u)-[:PARTICIPATED_IN]->(c)
            """
        def insert_batch(tx, rows):

            summary_counters = {
                "nodes_created": 0,
                "relationships_created": 0,
                "properties_set": 0,
            }
            for row in rows:
                result = tx.run(create_relationship_query, **row)
                counters = result.consume().counters

                summary_counters["nodes_created"] += counters.nodes_created
                summary_counters["relationships_created"] += counters.relationships_created
                summary_counters["properties_set"] += counters.properties_set
            
            logger.info(
                "Batch completed | rows=%d | nodes=%d | relationships=%d | properties=%d",
                len(rows),
                summary_counters["nodes_created"],
                summary_counters["relationships_created"],
                summary_counters["properties_set"],
            )

        with driver.session() as session:
        # Verify connectivity
            driver.verify_connectivity()
            logger.info("Connection established.")

        # Execute the query to create graph
        # Use database_="system" for administrative commands
        # Use IF NOT EXISTS to prevent errors if the database already exists

            
        
            try:
                with driver.session() as session:
                    batch = []
                    for _, r in df.iterrows():
                        batch.append({
                            "record_id": r["record_id"],
                            "user_id": r["user_id"],
                            "campaign": r["campaign"],
                            "message": r["message"],
                            "timestamp": datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S.%f").isoformat()
                        })

                        if len(batch) == 500:
                            session.execute_write(insert_batch, batch)
                            batch = []

                    if batch:
                        session.execute_write(insert_batch, batch)


            except Exception as e:
                logger.exception(f"An error occurred: {e}")



def create_user_campaign_relation():

    relationship_query = """

    MATCH (u:User)-[:PERFORMED]->(:Event)-[:PART_OF]->(c:Campaign)
    MERGE (u)-[r:TARGETED_BY]->(c)
    ON CREATE SET r.first_seen = datetime()
    """

    with GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "admin@1234")) as driver:
        with driver.session() as session:
            session.run(relationship_query)
    
    logger.info("Relationship created between users and campaigns")

def get_campaigns_user_is_part_of(tx, user_id):
    query = """
    MATCH (u:User {user_id: $user_id})-[:PARTICIPATED_IN]->(c:Campaign)
    RETURN c.campaign AS campaign
    ORDER BY campaign
    """
    return [r["campaign"] for r in tx.run(query, user_id=user_id)]

def main():
    try:

        logger.info("Script started")
        
        blobs  = gcs_ops_input.get_gcs_blobs(file_prefix=FILE_PREFIX)

        for blob in blobs:
            if blob.name.startswith(f"{FILE_PREFIX}/{FILE_NAME}") and blob.name.endswith(".csv"):
                logger.info(f"Starting processing for: {blob.name}")

                input_path = f"gs://{INPUT_BUCKET_NAME}/{blob.name}"

                logger.info(input_path)
                
                # Defining chunk size for avoiding OOM errors
                chunk_size = 500

                collection_instance_mongo = create_mongo_db_collection("marketing_db", "user_messages")

                client_instance_milvus, collection_milvus = milvus_db_ops()

                try:
                # Open output stream
                    with pd.read_csv(input_path, chunksize=chunk_size, dtype={"message": str}) as input_stream:
                        # Use enumerate to get a counter 'i' (0, 1, 2...) automatically
                        for chunk in input_stream:
                            
                            # Convert timestamp column to datetime
                            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
                            
                            # Add ingestion timestamp UTC (vectorized)
                            chunk["ingested_at"] = datetime.now(UTC)

                            # --------- Mongod db insert ----------#

                            records = chunk.to_dict(orient="records")

                            logger.info("Loading data into mongo db")
                            # Insert to collections setting ordered false
                            collection_instance_mongo.insert_many(records, ordered = False)

                            logger.info("data inserted to mongo db instance")

                            
                            logger.info("Starting encoding")
                            messages = chunk["message"]

                            valid_mask = messages.notna() & (messages.astype(str).str.strip() != "")
                            valid_texts = messages[valid_mask].tolist()

                            if valid_texts:
                                embeddings = model.encode(
                                    valid_texts,
                                    batch_size=64,
                                    show_progress_bar=False,
                                    convert_to_numpy=True
                                )
                            else:
                                embeddings = np.empty((0, EMBED_DIM), dtype=np.float32)

                            embedding_column = [
                                ZERO_VECTOR.copy() if isinstance(ZERO_VECTOR, np.ndarray) else list(ZERO_VECTOR)
                                for _ in range(len(chunk))
                            ]

                            emb_idx = 0
                            for i, is_valid in enumerate(valid_mask):
                                if is_valid:
                                    embedding_column[i] = embeddings[emb_idx]
                                    emb_idx += 1

                            chunk["text_embeddings"] = embedding_column

                            logger.info("Embedding added.")

                            # --------- Milvus db insert ---------- #
                            entities = []

                            for rid, uid, emb in zip(
                                chunk["record_id"],
                                chunk["user_id"],
                                chunk["text_embeddings"]
                            ):
                                entities.append({
                                    "record_id": str(rid),
                                    "user_id": str(uid) if pd.notna(uid) else "",
                                    "text_embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb
                                })

                            client_instance_milvus.insert(
                                collection_name=collection_milvus,
                                data=entities
                            )

                        client_instance_milvus.flush(collection_name=collection_milvus)

                        stats = client_instance_milvus.get_collection_stats(collection_name=collection_milvus)
                        print("Total entities:", stats["row_count"])

                        # Creating index for faster retrieval
                        # Prepare the index parameters
                        index_params = client_instance_milvus.prepare_index_params()
                        index_params.add_index(
                            field_name="text_embedding",
                            index_type="HNSW",
                            metric_type="COSINE",
                            params={
                                "M": 16,
                                "efConstruction": 200
                            }
                        )
                        client_instance_milvus.create_index(
                            collection_name=collection_milvus,
                            index_params=index_params
                        )
                


                    # -------- neo4j graph creating -----------------------------------
                    df = pd.read_csv(input_path)

                    # Create graph in neo4j based on dataframe columns
                    create_node()

                    create_basic_relationships(df)

                    create_user_campaign_relation()

                    # ---- Upload data from gcs to bigquery for analytics purpose --------------------------

                except Exception as e:
                    logger.exception(f"FAILED: {blob.name}. Error: {e}")

                # ---------- upload to bigquery table for analytics --------------------
                table_schema = [
                    bigquery.SchemaField("record_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("campaign", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE")
                ]

                # Create main table if it does not exist
                if bq_ops.table_exists(f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"):
                    pass
                else:
                    table = bigquery.Table("{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}", schema=table_schema)
                    table = bq_ops.bq_client.create_table(table)  # Make an API request.
                    logger.info(
                        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
                    )

                # Load data into staging
                bq_ops.create_native_table_using_gcs(f"{PROJECT_ID}.{DATASET_ID}.{STG_TABLE_ID}", input_path, table_schema)

                # Merge the result with main table
                merge_sql = util_func.read_sql_file_as_string(Path("./sql/merge_query.sql")).format(PROJECT_ID=PROJECT_ID, DATASET_ID=DATASET_ID, TABLE_ID=TABLE_ID, STG_TABLE_ID=STG_TABLE_ID)

                query_job = bq_ops.run_sql_query(merge_sql, job_config=None)

                print(f"Merge operation completed successfully. DML affected row count: {query_job.num_dml_affected_rows}")
                            
                logger.info(f"Completed loading data from {blob.name}")

    
    except GoogleCloudError as e:
        logger.exception("An error occurred during retrieving data")
        # The exception object contains the underlying HTTP status code and other details
        logger.exception(f"Status code: {e.code}")
        logger.exception(f"Details: {e.message}")
    
    except Exception as e:
        logger.exception(f"Exception occured, {e}")
        return "GCS connection failed.", 500

# Perform queries on dbs
def run_query_for_user_id(user_id):

    # Get user campaigns from node4j
    # for neo4j initializing client for running the query   
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            records = session.execute_read(get_campaigns_user_is_part_of, user_id)


    # Initializing client for milvus
    client = MilvusClient("/data/milvus_lite.db")
    unique_similar_users = get_most_similar_users(client, "user_data", user_id)

    # Get aggregation query results:


    logger.info(f"{user_id} participated in following campaigns: {[r.data() for r in records]}")

    logger.info(f"user is similar to following user : {unique_similar_users}")

if __name__ == "__main__":
    main()
    run_query_for_user_id("user_37")