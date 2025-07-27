from ibm_watsonx_ai import APIClient
import os
from dotenv import load_dotenv

load_dotenv()

credentials = {
    "url": os.environ["WATSONX_URL"],
    "apikey": os.environ["WATSONX_APIKEY"],
}
client = APIClient(
    credentials=credentials,
    project_id=os.environ["PROJECT_ID"]
)

print("Available methods of client.foundation_models:")
print([method for method in dir(client.foundation_models) if not method.startswith("_")])
