# test_create_index_fixed.py
import secrets
import cyborgdb_core as cyborgdb

API_KEY = "cyborg_de8158fd38be4b97a400d4712fa77e3d"

index_loc = cyborgdb.DBConfig(
    location="postgres",
    connection_string="host=localhost port=5432 dbname=cyborgdb user=cyborg password=cyborg123",
    table_name="cyborg_index"
)
config_loc = cyborgdb.DBConfig(
    location="postgres",
    connection_string="host=localhost port=5432 dbname=cyborgdb user=cyborg password=cyborg123",
    table_name="cyborg_config"
)
items_loc = cyborgdb.DBConfig(
    location="postgres",
    connection_string="host=localhost port=5432 dbname=cyborgdb user=cyborg password=cyborg123",
    table_name="cyborg_items"
)

print("DBConfig objects prepared. Initializing client...")
client = cyborgdb.Client(api_key=API_KEY, index_location=index_loc, config_location=config_loc, items_location=items_loc)

index_name = "test_index_from_script"
index_key = secrets.token_bytes(32)

# configure logger — use "info" (supported per docs) instead of "debug"
logger = cyborgdb.Logger.instance()
logger.configure(level="info", to_file=True, file_path="cyborg_index_creation.log")

print("Logger configured, creating index...")

try:
    # try creating index without embedding_model first (simpler)
    idx = client.create_index(index_name=index_name, index_key=index_key, embedding_model=None, logger=logger)
    print("Index created:", idx)
except Exception as e:
    print("Create index failed:", type(e), e)
    print("Check cyborg_index_creation.log and Postgres logs for details.")
    raise
