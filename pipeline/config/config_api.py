from typing import NamedTuple
import yaml

class Confing(NamedTuple):
    INPUT_BUCKET : str
    FILE_PREFIX : str
    PROJECT_ID : str
    DATASET_ID : str
    STG_TABLE_ID : str
    TABLE_ID : str
    FILE_NAME : str
    NEO4J_USER : str
    NEO4J_PASSWORD : str
    @staticmethod
    def load_yaml_config(path_val):
        with open(path_val, 'r') as file:
            data = yaml.safe_load(file)
        return(Confing(**data))


if __name__ == "__main__":
    pass