import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "outputs/output-{}.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


def ask_to_continue():
    return input('Continue recording?: [Y/N]').lower() == 'y'


frames = []

output_count = 0

while ask_to_continue():
    print("* recording")
    output_count += 1
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    output = WAVE_OUTPUT_FILENAME.format(output_count)
    print('Saving file: {}'.format(output))

    # save to disk
    wf = wave.open(output, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

stream.stop_stream()
stream.close()
p.terminate()
