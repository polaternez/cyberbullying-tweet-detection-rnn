import requests

url = "http://localhost:5000/predict_api"
data = {"input": [
    'In other words #katandandre, your food was crapilicious! #mkr', 
    'Why is #aussietv so white? #MKR #theblock #ImACelebrityAU #today #sunrise #studio10 #Neighbours #WonderlandTen #etc',
    'Yea fuck you RT @therealexel: IF YOURE A NIGGER FUCKING UNFOLLOW ME, FUCKING DUMB NIGGERS.',
    'Bro. U gotta chill RT @CHILLShrammy: Dog FUCK KP that dumb nigger bitch lmao'
]}
r = requests.post(url, json=data)

print(r.text)
