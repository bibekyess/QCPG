import requests

API_URL = "https://api-inference.huggingface.co/models/bibekyess/qcpg-parabk2-t5-base"
API_TOKEN='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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "We will do very good in the final project of this class.",
})
print(output)