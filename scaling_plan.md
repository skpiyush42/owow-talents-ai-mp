Following points should be considered while scaling up this pipeline
---------------

#### 1. Deploy the docker container to GKE or cloud run or some similar cloud service
#### 2. Use Workload Identity Federation available if deploying on cloud insted of using the Service account json.
#### 3. Use Milvus standalone database for production grade workloads. It makes uses of minio service as object store for stores segments & index filesas and etcd service metadata, schema, cluster state as a full fledged vector database
#### 4. Use graphana for pipeline monitoring.
#### 5. As data grows in size we can store the vector embeddings in a seperate collection for each embedding model in milvus 