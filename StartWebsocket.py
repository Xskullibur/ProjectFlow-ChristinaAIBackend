import asyncio
import websockets

import redis

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from Crypto.Hash import SHA
from Crypto import Random
from base64 import b64decode, b64encode
import hmac
import hashlib

import os
import json

import Predict as predict
from VoiceDataset import pad
import numpy as np

import speech_recognition as sp_reco

# voice recordings path
RECORDING_PATH = r'G:\NYP\EADP\Storage'

# read private key
privateKey = RSA.importKey(open("misc/rsa.private", "r").read())
cipher = PKCS1_v1_5.new(privateKey)

# Read salt and key binary files
salt = None
key = None
with open('misc/salt.bin', mode='rb') as file:
    salt = file.read()
with open('misc/key.bin', mode='rb') as file:
    key = file.read()

r = redis.Redis(host='192.168.99.100', port=6379, db=0)

# load model
siamese = predict.load_model()


async def handle(websocket, path):
    # perform handshake with client
    room = await handshake(websocket, path)

    # connect to ms sql
    cursor = connectToDB()

    # fetch all students belonging to the team
    teamID = room['teamID']
    roomID = room['roomID']

    try:
        await predictionHandle(websocket, path, teamID, roomID, cursor)
    finally:
        print('Connection ended')


async def handshake(websocket, path):
    password = await websocket.recv()
    print(password)
    r_password = b64decode(password)

    dsize = SHA.digest_size
    sentinel = Random.new().read(15 + dsize)

    decrypted = bytearray(0)
    for i in range(0, len(r_password), 256):
        data = cipher.decrypt(r_password[i:i + 256], sentinel)
        decrypted.extend(data)
    decrypted = bytes(decrypted)

    passwordWithSalt = exclusiveOR(decrypted, salt)

    hashedPasswordWithSalt = hmac.new(key, msg=passwordWithSalt, digestmod=hashlib.sha256).digest()

    value = r.get(b64encode(hashedPasswordWithSalt))
    room = json.loads(value)
    return room


async def predictionHandle(websocket, path, teamID, roomID, cursor):
    # read wav files from previous recordings

    speakers = getSpeakers(cursor, teamID)
    recordings = {}
    for speaker in speakers:
        print(speaker.UserId + ', ' + speaker.studentId + ':')
        recordings[speaker.UserId] = []
        for recording in getVoiceRecordingFromSpeaker(cursor, speaker.UserId):
            sr, data = predict.load_wav(os.path.join(RECORDING_PATH, recording.recordingName))
            # pad data
            data = np.array(pad(data, predict.length)[:predict.length]).reshape(-1, predict.length, 1)
            recordings[speaker.UserId].append((recording.recordingName, data))
            print('{:>25s}'.format(recording.recordingName))

    connection = True
    while connection:
        instr = await websocket.recv()
        if instr == 'BLOB':
            # read bytes
            blob = await websocket.recv()

            audio_file = 'misc/{}.ogg'.format(teamID)
            audio_file_wav = 'misc/{}.wav'.format(teamID)
            with open(audio_file, 'wb') as file:
                file.write(blob)
                print('Write to disk')
                convert(teamID)
                sr, data = predict.load_wav(audio_file_wav)

                # pad data
                data = np.array(pad(data, predict.length)[:predict.length])

                predicted_speaker = speakers[0]
                prediction_speaker_value = 0
                for speaker in speakers:
                    for recording in recordings[speaker.UserId]:
                        prediction = 1.0 - siamese.predict([data.reshape(-1, predict.length, 1), recording[1]])[0][0]
                        if prediction > prediction_speaker_value:
                            predicted_speaker = speaker
                            prediction_speaker_value = prediction
                print(prediction_speaker_value)
                print(predicted_speaker)

                userID = predicted_speaker[0]

                transcript = ''
                no_speaker = False
                try:
                    transcript = recognize(audio_file_wav)

                    # insert to db
                    insertTranscript(cursor, userID, transcript, roomID)
                except sp_reco.UnknownValueError:
                    no_speaker = True

                print("Transcription: " + transcript)
                if not no_speaker:
                    # send back to client
                    username = getUsernameFromUserID(cursor, userID).UserName
                    await websocket.send('{{"predicted_speaker": "{}", "transcript": "{}"}}'
                                         .format(username, transcript))
                else:
                    await websocket.send('{}')



        else:
            connection = False
            print('Ending connection')


def exclusiveOR(ba1, ba2):
    if len(ba1) == len(ba2):
        return bytes([a ^ b for (a, b) in zip(ba1, ba2)])
    else:
        raise ValueError('Invalid length')


def recognize(wav_file):
    recorder = sp_reco.Recognizer()
    with sp_reco.WavFile(wav_file) as source:
        audio = recorder.record(source)
        with open('private/projectflow-264706-9ac1a4140d2a.json', 'r') as file:
            transcript = recorder.recognize_google_cloud(audio, credentials_json=file.read())
            return transcript


import pyodbc


def connectToDB():
    server = '192.168.99.100,1433'
    db = 'ProjectFlow'
    username = 'sa'
    password = 'Password123'
    cnxn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' +
        db + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    return cursor


def getSpeakers(cursor, teamID):
    cursor.execute('SELECT t.UserId, studentId FROM TeamMember t INNER JOIN Student '
                   's ON t.UserId = s.UserId WHERE teamID = ?', teamID)
    rows = cursor.fetchall()
    return rows


def getVoiceRecordingFromSpeaker(cursor, userID):
    cursor.execute('SELECT recordingName FROM VoiceRecording WHERE UserId = ?', userID)
    rows = cursor.fetchall()
    return rows


def getUsernameFromUserID(cursor, userID):
    cursor.execute('SELECT UserName FROM aspnet_Users WHERE UserId = ?', userID)
    row = cursor.fetchone()
    return row


def insertTranscript(cursor, userID, transcript, roomID):
    cursor.execute('INSERT INTO Transcript VALUES (?, ?, ?)', transcript, userID, roomID)
    cursor.commit()


import subprocess


def convert(teamID):
    '''
    Convert file located in 'misc/<teamID>.ogg' into 'misc/<teamID>.wav'
    '''
    subprocess.call('ffmpeg/ffmpeg -i misc/{0}.ogg -y misc/{0}.wav'.format(teamID))


start_server = websockets.serve(handle, 'localhost', 9000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
