#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from pathlib import Path
import pyaudio
import sys
import time
from scipy.io import wavfile


from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


MOZILLA_TTS_DIR = (
    "/home/zeroos/PythonEnv/mozillaTTS2/lib/python3.6/site-packages/TTS"
)

def main():
    # load model manager
    path = Path(MOZILLA_TTS_DIR) / ".models.json"

    manager = ModelManager(path)

    model_name = "tts_models/en/ljspeech/tacotron2-DDC"

    model_path, config_path, model_item = manager.download_model(model_name)
    vocoder_name = model_item["default_vocoder"]

    vocoder_path, vocoder_config_path, _ = (
        manager.download_model(vocoder_name)
    )
    use_cuda = True

    speakers_file_path = None
    encoder_path = None
    encoder_config_path = None

    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda,
    )

    FIFO = '/tmp/ddc_synthesizer'
    try:
        os.mkfifo(FIFO)
    except:
        print(f"Unable to create fifo. Maybe '{FIFO}' already exists?")
        raise

    try:
        player = WavePlayer(add_noise=True)
        while True:
            print("Waiting for text")
            with open(FIFO) as fifo:
                while True:
                    text = fifo.read()
                    if len(text) == 0:
                        print("Writer closed")
                        break
                    print()
                    print('Read: "{0}"'.format(text))
                    print()

                    sentences = text.split(',')
                    for sentence in sentences:
                        sentence = sentence.replace("i l s", "i el es")
                        print(" > Text: {}".format(sentence))

                        # kick it
                        wav = synthesizer.tts(sentence)
                        player.play_wav(wav)

        print("end of read program")
    finally:
        os.remove(FIFO)
        player.close()

    # save the results
    # out_path = "/tmp/out.wav"
    # print(" > Saving output to {}".format(out_path))
    # synthesizer.save_wav(wav, out_path)


class WavePlayer:
    def __init__(self, add_noise=False):
        #instantiate PyAudio
        self.p = pyaudio.PyAudio()
        #open stream
        sampwidth = 2
        self.stream = self.p.open(
            format = self.p.get_format_from_width(sampwidth),
            channels = 1,
            rate = 22050,
            output = True,
        )

        if add_noise:
            NOISE_TYPE = 1
            if NOISE_TYPE == 1:
                _, self.noise_data = wavfile.read('./static.wav')
                self.noise_data = self.noise_data/30000
            else:
                _, self.noise_data = wavfile.read('./static2.wav')
                self.noise_data = self.noise_data[200000:]/100000
        else:
            self.noise_data = None

    def play_wav(self, wav):
        wav = np.array(wav + [0]*100000)

        if self.noise_data is not None:
            wav += self.noise_data[0:len(wav)]
        # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav_norm = wav * (52767 / max(0.01, np.max(np.abs(wav))))
        wav_data = wav_norm.astype(np.int16)

        #play stream
        self.stream.write(wav_data)


    def close(self):
        #stop stream
        self.stream.stop_stream()
        self.stream.close()

        #close PyAudio
        self.p.terminate()



if __name__ == "__main__":
    main()
