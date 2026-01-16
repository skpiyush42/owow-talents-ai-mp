# 🏗️ System Architecture

This project implements an **end-to-end AI-driven marketing personalization platform** using a containerized ETL pipeline and a read-only FastAPI query service. The system ingests raw data from Google Cloud Storage, enriches it using NLP and graph modeling, and exposes recommendation and analytics APIs.

---

## High-Level Architecture

```
Google Cloud Storage (CSV files)
            |
            v
+----------------------------------+
|  ETL App (owow-talents-ai-mp)     |
|  - Data ingestion & validation   |
|  - Sentence embeddings           |
|  - Graph creation (Neo4j)        |
|  - Vector storage (Milvus Lite)  |
|  - Analytics load (BigQuery)     |
+----------------------------------+
     |        |        |        |
     v        v        v        v
 MongoDB   Milvus   Neo4j   BigQuery
 (raw)     Lite     Graph   Warehouse
                                 |
                                 v
                     +------------------------+
                     | FastAPI Query Service  |
                     | (owow-query-api)       |
                     +------------------------+
```

---

## Components Overview

### 1. Data Source – Google Cloud Storage (GCS)

**Purpose**
- Acts as the ingress layer for raw CSV files
- Stores marketing and user-event data

**Usage in Code**
- Files are read in chunks (`chunksize=500`) to avoid memory pressure
- Filtered by prefix and filename before processing

---

### 2. ETL Application (`owow-talents-ai-mp`)

**Deployment Model**
- Runs as a one-shot container
- Executes the full pipeline
- Exits with code `0` on success

**Key Responsibilities**

#### a. MongoDB – Raw Data Store
- Stores raw ingested records
- Adds `ingested_at` UTC timestamp
- Provides auditability and replay capability

#### b. NLP Embedding Layer
- Uses `SentenceTransformer(all-MiniLM-L6-v2)`
- Embedding dimension: `384`
- Handles empty messages using zero vectors

#### c. Milvus Lite – Vector Database
- Persistent local DB (`/data/milvus_lite.db`)
- Stores:
  - `record_id` (primary key)
  - `user_id`
  - `text_embedding`
- Indexed with COSINE similarity
- Used for user similarity recommendations

#### d. Neo4j – Graph Database
**Nodes**
- `User`
- `Event`
- `Campaign`

**Relationships**
- `(:User)-[:PERFORMED]->(:Event)`
- `(:Event)-[:PART_OF]->(:Campaign)`
- `(:User)-[:PARTICIPATED_IN]->(:Campaign)`
- Derived:
  - `(:User)-[:TARGETED_BY]->(:Campaign)`

**Purpose**
- Behavioral modeling
- Campaign attribution
- Graph traversals

#### e. BigQuery – Analytics Warehouse
- Data loaded via GCS native load
- Staging → merge into final table
- Supports long-term analytics and KPIs

---

## 3. Persistent Datastores

| System      | Purpose                  |
|------------|--------------------------|
| MongoDB    | Raw & audit data         |
| Milvus Lite| Vector similarity search |
| Neo4j      | Graph relationships      |
| BigQuery   | Analytical queries       |

All stateful services persist data via Docker volumes.

---

## 4. FastAPI Query Service (`owow-query-api`)

**Startup Rule**
- Starts **only after ETL completes successfully**
- Enforced using Docker Compose:
```yaml
depends_on:
  etl-app:
    condition: service_completed_successfully
```

**Design Principles**
- Read-only
- Stateless
- Low-latency
- No recomputation

---

## API Endpoints

### 🔹 Vector Recommendations (Milvus)
```
GET /recommendations/similar-users
```

### 🔹 Graph Queries (Neo4j)
```
GET /graph/user-campaigns
```

### 🔹 Analytics (BigQuery)
```
GET /analytics/user-engagement
```

---

## 5. Orchestration – Docker Compose

**Startup Order**
```
MongoDB
  ↓
Neo4j
  ↓
ETL App (runs once, exits)
  ↓
FastAPI Query Service
```

**Key Guarantees**
- FastAPI never starts on partial or failed data
- ETL failures block API startup
- Clean separation of compute vs serving

---

## Architectural Strengths

### ✔ Separation of Concerns
- ETL handles compute and enrichment
- FastAPI handles queries only
- Storage systems are purpose-built

### ✔ Failure Isolation
- ETL and API are decoupled
- Reruns are safe and deterministic

### ✔ Scalability
- ETL can be scheduled (cron / Airflow)
- FastAPI can scale horizontally
- Datastores evolve independently

---

## Future Enhancements

- Scheduled ETL execution
- Graph-based recommendation ranking
- Authentication & rate limiting
- Observability (metrics, tracing)
- Feature store abstraction
