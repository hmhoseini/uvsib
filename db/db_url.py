import os
from dotenv import load_dotenv
from uvsib.workflows import settings

keys = ["USER", "PASSWORD", "DB"]
dotenv_path = os.path.join(settings.run_dir, ".env")
ENV_NAME="DEV"
load_dotenv(dotenv_path=dotenv_path)
config = {key: os.getenv(f"{ENV_NAME}_{key}") for key in keys}
user = config["USER"]
password = config["PASSWORD"]
db = config["DB"]
host = os.getenv("HOST", "localhost")
port = os.getenv("PORT", "5432")
DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
