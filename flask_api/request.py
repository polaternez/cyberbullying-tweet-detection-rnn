import requests
from requests.exceptions import RequestException


URL = "http://localhost:5000/predict_api"

payload = {
    "input": [
        'In other words #katandandre, your food was crapilicious! #mkr', 
        'Why is #aussietv so white? #MKR #theblock #ImACelebrityAU #today #sunrise #studio10 #Neighbours #WonderlandTen #etc',
        'Yea fuck you RT @therealexel: IF YOURE A NIGGER FUCKING UNFOLLOW ME, FUCKING DUMB NIGGERS.',
        'Bro. U gotta chill RT @CHILLShrammy: Dog FUCK KP that dumb nigger bitch lmao'
    ]
}

try:
    response = requests.post(URL, json=payload, timeout=30)

    # Raise an exception for HTTP errors (4xx, 5xx)
    response.raise_for_status()

    # Parse JSON response
    result = response.json()

    print("Prediction Result:")
    print(result)

except RequestException as e:
    print(f"Request failed: {e}")

except ValueError:
    print("Server returned an invalid JSON response.")


