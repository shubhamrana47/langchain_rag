import requests 

response=requests.get("https://official-joke-api.appspot.com/random_joke");
print(response)