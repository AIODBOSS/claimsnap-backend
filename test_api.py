import requests

url = "http://127.0.0.1:5000/api/claims"
video_path = "test_video.webm" # Change this to any small video file you have

files = {
    'video': open(video_path, 'rb')
}

data = {
    'claimType': 'vehicle',
    'description': 'Test claim for AI analysis',
    'policyNumber': 'HI-2026-TEST',
    'incidentDate': '2026-05-05',
    'contactPhone': '555-0199'
}

print(f"🚀 Sending claim to {url}...")
response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
try:
    print("Response Body:", response.json())
except:
    print("Response is not JSON. HTML Error received:")
    print(response.text[:500]) # Print the first 500 characters of the error