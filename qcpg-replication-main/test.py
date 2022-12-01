import requests

API_URL = "https://api-inference.huggingface.co/models/bibekyess/qcpg-parabk2-mt5"
API_TOKEN='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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "COND_SEMANTIC_SIM_80 COND_LEXICAL_DIV_30 COND_SYNTACTIC_DIV_50Is this going to work or what are we doing here?",
})
print(output)