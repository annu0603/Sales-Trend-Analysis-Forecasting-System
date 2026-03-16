import os
from google import genai as gemini
from dotenv import load_dotenv

class init_models:

    def __init__(self):
        load_dotenv()
        self.gemini_client = gemini.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def gemini_model(self, model_name, contents):
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=contents)
        return response.text
