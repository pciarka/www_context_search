import os
from dotenv import load_dotenv
load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])