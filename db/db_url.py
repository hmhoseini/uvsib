import os
from dotenv import load_dotenv

current = os.path.dirname(os.path.realpath(__file__))
keys = ["USER", "PASSWORD", "DB"]
dotenv_path=os.path.join(current, ".env")
ENV_NAME="DEV"
load_dotenv(dotenv_path=dotenv_path)
config = {key: os.getenv(f"{ENV_NAME}_{key}") for key in keys}
user = config["USER"]
password = config["PASSWORD"]
db = config["DB"]
host = os.getenv("HOST", "localhost")
port = os.getenv("PORT", "5432")
DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
