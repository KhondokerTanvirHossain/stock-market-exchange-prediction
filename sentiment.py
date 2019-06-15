import requests
from textblob import TextBlob

url = ('https://newsapi.org/v2/everything?q=%20google%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
#print (response.json())
parsed_json = response.json()
#print(parsed_json['status'])
array = parsed_json['articles']
polarity = 0.0;
count = 0;
for i in array:
    #print(i['description'])
    blob = TextBlob(i['description'])
    count = count + 1
    polarity = polarity + blob.sentiment.polarity
polarity = polarity / count
print(polarity)    

url = ('https://newsapi.org/v2/everything?q=%20Apple%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
#print (response.json())
response = requests.get(url)
#print (response.json())
parsed_json = response.json()
#print(parsed_json['status'])
array = parsed_json['articles']
polarity = 0.0;
count = 0;
for i in array:
    #print(i['description'])
    blob = TextBlob(i['description'])
    count = count + 1
    polarity = polarity + blob.sentiment.polarity
polarity = polarity / count
print(polarity)    

url = ('https://newsapi.org/v2/everything?q=%20Microsoft%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
#print (response.json())
response = requests.get(url)
#print (response.json())
parsed_json = response.json()
#print(parsed_json['status'])
array = parsed_json['articles']
polarity = 0.0;
count = 0;
for i in array:
    #print(i['description'])
    blob = TextBlob(i['description'])
    count = count + 1
    polarity = polarity + blob.sentiment.polarity
polarity = polarity / count
print(polarity)    

url = ('https://newsapi.org/v2/everything?q=%20IBM%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
#print (response.json())
response = requests.get(url)
#print (response.json())
parsed_json = response.json()
#print(parsed_json['status'])
array = parsed_json['articles']
polarity = 0.0;
count = 0;
for i in array:
    #print(i['description'])
    blob = TextBlob(i['description'])
    count = count + 1
    polarity = polarity + blob.sentiment.polarity
polarity = polarity / count
print(polarity)    

url = ('https://newsapi.org/v2/everything?q=%20amazon%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
#print (response.json())
response = requests.get(url)
#print (response.json())
parsed_json = response.json()
#print(parsed_json['status'])
array = parsed_json['articles']
polarity = 0.0;
count = 0;
for i in array:
    #print(i['description'])
    blob = TextBlob(i['description'])
    count = count + 1
    polarity = polarity + blob.sentiment.polarity
polarity = polarity / count
print(polarity)