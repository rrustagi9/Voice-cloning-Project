# Voice cloning project 
This project us based on the repository which is an implementation of Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS) with a vocoder that works in real-time. Feel free to check my thesis if you're curious or if you're looking for info I haven't documented. Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.
Datasets

For playing with the toolbox alone, I only recommend downloading LibriSpeech/train-clean-100. Extract the contents as <datasets_root>/LibriSpeech/train-clean-100 where <datasets_root> is a directory of your choosing. Other datasets are supported in the toolbox, see here. You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.


# setup
1. Install Requirements
Python 3.6 or 3.7 is needed to run the toolbox.

Install PyTorch (>=1.0.1).
Install ffmpeg.
Run pip install -r requirements.txt to install the remaining necessary packages.
2. Download Pretrained Models
Download the latest here.

3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

python demo_cli.py

If all tests pass, you're good to go.

4. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading LibriSpeech/train-clean-100. Extract the contents as <datasets_root>/LibriSpeech/train-clean-100 where <datasets_root> is a directory of your choosing. Other datasets are supported in the toolbox, see here. You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.
