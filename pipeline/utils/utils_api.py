import logging
from google.api_core import exceptions
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ds-cdr-portalreportin-manual-gcs-logger")
logger.setLevel(logging.INFO)

class Utils:
    
    def _init__(self):
        pass
    
    @staticmethod
    def read_sql_file_as_string(filename):
        """
        Reads the entire content of a file into a single string.

        Args:
            filename (str): The path to the SQL file.

        Returns:
            str: The content of the SQL file as a single string.
        """
        try:
            # Open and read the file in text mode ('r') with 'utf-8' encoding
            with open(filename, 'r', encoding='utf-8') as file:
                sql_script = file.read()
            return sql_script
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

if __name__ == "__main__":
    pass