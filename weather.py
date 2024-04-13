import requests

api_key = "f0a55f9c59cd99af20573d7b91a14b1f"
city = "London"
base_url = "https://api.openweathermap.org/data/2.5/weather?"

complete_url = base_url + "appid=" + api_key + "&q=" + city
response = requests.get(complete_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()
    main = data['main']
    temperature = main['temp']
    description = data['weather'][0]['description']

    print(f"Current weather in {city}:")
    print(f"Temperature: {temperature} Kelvin")
    print(f"Description: {description}")
else:
    print("Error: Could not retrieve weather data")
    print(response.json())
