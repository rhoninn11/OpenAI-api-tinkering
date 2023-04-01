import os
import openai
import unicodedata

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

messages = [
        {"role": "system", "content": "Niech twoje odpowiedzi będą w formacie plików json"},
        {"role": "user", "content": "Przedstaw mi potencjalną strukturę dużej firmy o nazwie PCSS"}  
    ]

result = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages
)

response = result["choices"][0]
message = response["message"] 
to_print = unicodedata.normalize('NFKD', message["content"]).encode('utf-8').decode('utf-8')
print(to_print)