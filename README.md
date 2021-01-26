Generating new Bach music using Long Short-Term Memory (LSTM) recurrent neural networks and PyTorch. 

## Data
A collection of Bach MIDI files contained in the `data` folder are used for training. Since the MIDI files must somehow be turned into a format that can be fed into the LSTM, the `data` folder also contains encoded versions of the MIDI files. Bach MIDI files are from David Grossman's awesome website at http://www.jsbach.net/midi/index.html.

### How are the MIDI files encoded?
MIDI files are converted to CSV using `py-midicsv` (https://pypi.org/project/py-midicsv/), after which we convert the MIDI representation into one long string where all notes are quantized to 64th notes and converted to an ASCII representation. If you want to convert your own MIDI files, you can do this as follows:  

```
python miditocsv.py -i midi_file.mid -o out.txt
python preprocess.py -i out.txt -o midi_file_encoded.txt
```

If you want to undo the encoding, or simply turn the LSTM generated note sequences into MIDI so that you can actually give them a listen, run the following:  

```
python deprocess.py -i midi_file_encoded.txt -o out.txt --tempo 250000
python csvtomidi.py -i out.txt -o midi_file.mid
```

Note that the encoding is not lossless, since some data from the original MIDI file is discarded (e.g. track title, tempo). 
