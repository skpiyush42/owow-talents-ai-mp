from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
from google.api_core.exceptions import NotFound
import logging
import sys

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ds-cdr-portalreportin-manual-bigquery_module-logger")
logger.setLevel(logging.INFO)

class BigqueryOperations:
    # Class attributes
    bq_client = bigquery.Client()

    def __init__(self, project_id):
        self.projet_id = project_id
    
    def run_sql_query(self, sql_query, job_config):
        """
        Function to run sql in bigquery export
        
        :param sql_query: query string
        :param job_config: job configuration
        """
        try:
            query_job = self.bq_client.query(sql_query, job_config=job_config)
            # Wait for the query to finish and get results
            query_job.result()

            return query_job

        except GoogleCloudError:
            raise GoogleCloudError

        except Exception:
            raise Exception

    
    def export_to_table(self, sql_query, destination_table):
        """
        Function to export data to bigquery table
        
        :param sql_query: SQL query string
        :param destination_table: destination table id
        """
        try:
            job_config = bigquery.QueryJobConfig(
                destination=destination_table,
                write_disposition="WRITE_TRUNCATE" # Overwrites table if it exists
            )
            export_job = self.bq_client.query(sql_query, job_config=job_config)
            
            logger.info(f"Bigquery job started to export data to {destination_table} table")

            export_job.result()

            logger.info(f"{destination_table} exported successfully")

        except GoogleCloudError:
            raise GoogleCloudError

        except Exception:
            raise Exception
    

    def export_to_gcs(self, sql_query, destination_table, destination_uri):
        
        try:
            job_config = bigquery.QueryJobConfig(
                    destination=destination_table,
                    write_disposition="WRITE_TRUNCATE" # Overwrites table if it exists
            )

            self.bq_client.query(sql_query, job_config=job_config).result()
            extract_job = self.bq_client.extract_table(destination_table, destination_uri)

            logger.info(f"Bigquery job started to export {destination_table} table data to {destination_uri} bucket uri")

            extract_job.result()

            logging.info(f"{destination_table} exported successfully to gcs url : {destination_uri}")

        except GoogleCloudError:
            raise GoogleCloudError

        except Exception:
            raise Exception

    def create_native_table_using_gcs(self, bq_table_id, source_uri, schema):
    
        try:
            # configure load job
            job_config = bigquery.LoadJobConfig(
                schema = schema,
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,      # Skip header if it exists
                autodetect=True,          # Infers schema (types like INT, STRING) automatically
                write_disposition="WRITE_TRUNCATE"
            )

            # Start the load job from GCS
            load_job = self.bq_client.load_table_from_uri(
                source_uri, 
                bq_table_id, 
                job_config=job_config
            )

            logger.info(f"Bigquery job started create {bq_table_id} table")
            
            # Wait for the job to complete
            load_job.result()  

            # Verify the results
            destination_table = self.bq_client.get_table(bq_table_id)
            logger.info(f"Successfully loaded {destination_table.num_rows} rows into {bq_table_id}.")

            return bq_table_id

        except GoogleCloudError:
            raise GoogleCloudError

        except Exception:
            raise Exception
        
    def table_exists(self, table_id):
        """Checks if a BigQuery table exists."""
        try:
            self.bq_client.get_table(table_id) # Make an API request.
            return True
        except NotFound:
            return False
        

if __name__ == "__main__":
    pass


