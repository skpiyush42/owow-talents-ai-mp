from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from neo4j import GraphDatabase
from pymilvus import MilvusClient
from google.cloud import bigquery
import numpy as np
import os
import logging

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("owow-query-service")

# ---------------- Config ----------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin@1234")

MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "/data/milvus_lite.db")
MILVUS_COLLECTION = "user_data"

BQ_PROJECT = os.getenv("BQ_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

# ---------------- Clients ----------------
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

milvus_client = MilvusClient(MILVUS_DB_PATH)
bq_client = bigquery.Client(project=BQ_PROJECT)

# ---------------- App ----------------
app = FastAPI(
    title="OWOW Recommendation & Analytics API",
    version="1.0.0"
)

# ---------------- Schemas ----------------
class SimilarUser(BaseModel):
    user_id: str
    similarity_score: float

class CampaignResponse(BaseModel):
    campaigns: list[str]

class AnalyticsResponse(BaseModel):
    user_id: str
    total_events: int
    first_event_time: str | None
    last_event_time: str | None

# ============================================================
# 1. MILVUS — VECTOR RECOMMENDATIONS
# ============================================================
@app.get(
    "/recommendations/similar-users",
    response_model=list[SimilarUser],
    summary="Recommend similar users using vector embeddings"
)
def recommend_similar_users(
    user_id: str = Query(...),
    limit: int = Query(5, ge=1, le=20)
):
    records = milvus_client.query(
        collection_name=MILVUS_COLLECTION,
        filter=f'user_id == "{user_id}"',
        output_fields=["text_embedding"]
    )

    if not records:
        raise HTTPException(404, f"No embeddings found for user_id={user_id}")

    vectors = np.array(
        [r["text_embedding"] for r in records],
        dtype=np.float32
    )

    query_vector = vectors.mean(axis=0)

    results = milvus_client.search(
        collection_name=MILVUS_COLLECTION,
        data=[query_vector.tolist()],
        anns_field="text_embedding",
        limit=limit,
        filter=f'user_id != "{user_id}"',
        output_fields=["user_id"],
        search_params={"metric_type": "COSINE"}
    )

    seen = set()
    response = []

    for hit in results[0]:
        uid = hit["entity"]["user_id"]
        if uid not in seen:
            seen.add(uid)
            response.append({
                "user_id": uid,
                "similarity_score": hit["distance"]
            })

    return response

# ============================================================
# 2. NEO4J — GRAPH RELATIONSHIPS
# ============================================================
@app.get(
    "/graph/user-campaigns",
    response_model=CampaignResponse,
    summary="Fetch campaigns a user participated in (graph traversal)"
)
def get_user_campaigns(
    user_id: str = Query(...)
):
    query = """
    MATCH (u:User {user_id: $user_id})-[:PARTICIPATED_IN]->(c:Campaign)
    RETURN c.campaign AS campaign
    ORDER BY campaign
    """

    with neo4j_driver.session() as session:
        campaigns = [
            r["campaign"] for r in session.run(query, user_id=user_id)
        ]

    return {"campaigns": campaigns}

# ============================================================
# 3. BIGQUERY — ANALYTICS / KPIs
# ============================================================
@app.get(
    "/analytics/user-engagement",
    response_model=AnalyticsResponse,
    summary="User engagement analytics from BigQuery"
)
def get_user_engagement_analytics(
    user_id: str = Query(...)
):
    sql = f"""
    SELECT
      COUNT(1) AS total_events,
      MIN(timestamp) AS first_event_time,
      MAX(timestamp) AS last_event_time
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE user_id = @user_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
        ]
    )

    result = list(
        bq_client.query(sql, job_config=job_config).result()
    )

    if not result:
        raise HTTPException(404, "No analytics found")

    row = result[0]

    return {
        "user_id": user_id,
        "total_events": row.total_events,
        "first_event_time": str(row.first_event_time),
        "last_event_time": str(row.last_event_time)
    }
