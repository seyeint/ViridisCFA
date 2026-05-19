import os
from dotenv import load_dotenv
from edgar import set_identity
from openai import OpenAI

# Load environment variables
load_dotenv()

# EDGAR configuration
EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY")
if EDGAR_IDENTITY:
    set_identity(EDGAR_IDENTITY)
    # Print only once or when required, but let's keep it quiet or log it
    # print(f"Using EDGAR Identity: {EDGAR_IDENTITY}")

# OpenAI configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize client
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

def get_openai_client():
    global client
    if client is None and OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY)
    return client
