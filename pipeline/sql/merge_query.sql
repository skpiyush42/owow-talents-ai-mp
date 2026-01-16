MERGE INTO `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` T
USING `{PROJECT_ID}.{DATASET_ID}.{STG_TABLE_ID}` S
ON T.record_id = S.record_id
WHEN MATCHED THEN
  UPDATE SET T.user_id = S.user_id,
  T.text = S.text,
  T.timestamp = S.timestamp
WHEN NOT MATCHED THEN
  INSERT (record_id, user_id, text, timestamp) 
  VALUES (S.record_id, S.user_id, S.text, S.timestamp)