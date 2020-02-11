import speech_recognition as sr


r = sr.Recognizer()

list = sr.Microphone.list_microphone_names()

with sr.Microphone() as source:
    print('Listening')
    audio = r.listen(source, phrase_time_limit=3)
    with open('private/projectflow-264706-9ac1a4140d2a.json', 'r') as file:
        print(r.recognize_google_cloud(audio, credentials_json=file.read()))
