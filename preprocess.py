""" 
Preprocess a midi file converted to txt using py_midicsv, producing a list of
note vectors, one for each predefined time step. 
"""

import csv
import re
import argparse
import string
import pickle
import json
from pathlib import Path

import py_midicsv as pm


def encode_seq(seq):
    """ Encode a sequence of notes into a more compact representation
    consisting of an ascii character for each note. """
    res = ""
    for note in seq:
        res += note_to_enc[int(note)]
    return res


def timesteps_from_ticks(ticks, timestep_ticks):
    assert timestep_ticks != 0, "timestep_ticks should not be 0."
    return round(ticks / timestep_ticks)


def append_or_update(dic, key, val):
    if key in dic:
        dic[key].append(val)
    else:  # create new entry
        dic.update({key: [val]})
    return dic


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile')
parser.add_argument('-o', '--outfile')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile

'''
# Write note encodings to disk. 
codes = (string.digits + string.ascii_letters +
         string.punctuation).replace('\\', '').replace('"', '').replace('\'', '')[:88]
note_range = list(range(21, 109))  # midi notes ranging from A0 to C7
note_to_enc = dict(zip(note_range, codes))
with open('encoding.json', 'w') as f:
    json.dump(note_to_enc, f)
'''

# Load note encoding dictionary.
with open('encoding.json', 'r') as f:
    note_to_enc = json.load(f)
note_to_enc = {int(k): v for k, v in note_to_enc.items()}
enc_to_note = {v: k for k, v in note_to_enc.items()}


# TODO: check whether time signature should be taken into account.

with open(infile, newline='') as f:
    dialect = csv.excel()
    dialect.skipinitialspace = True
    reader = csv.reader(f, delimiter=',', dialect=dialect)
    note_vecs = []
    stack = {}
    state = 0
    timestep_ticks = 0
    for line in reader:
        if state == 0:
            if line[2] == 'Header':
                div = int(line[5])  # no. of clock pulses per quarter note
                timestep_ticks = div / 16  # 1 timestep = 64th note
                state = 1
        elif state == 1: # search for start of track
            if line[2] == 'Note_on_c': # new track
                if note_vecs == []:
                    note_vecs.append([])
                note = int(line[4])
                ticks = int(line[1])  
                t = timesteps_from_ticks(ticks, timestep_ticks)
                # Keys in stack have list values bcs it is possible to play the
                # same note twice without releasing it.
                stack = append_or_update(stack, note, t)
                state = 2
        elif state == 2: # inside track
            if line[2] == 'Note_off_c' or (line[2] == 'Note_on_c' and int(line[5]) == 0):
                note = int(line[4])
                ticks = int(line[1])
                t = timesteps_from_ticks(ticks, timestep_ticks)
                if (n := len(note_vecs)) < t: # add new note vectors
                    note_vecs += [[] for _ in range(t-n)]
                t_start = stack.get(note)
                if t_start is not None:
                    t_start = t_start.pop()
                    for tick in range(t_start, t):
                        note_vecs[tick].append(note)
                    if stack[note] == []:
                        del stack[note]
            elif line[2] == 'Note_on_c':
                note = int(line[4])
                ticks = int(line[1])
                t = timesteps_from_ticks(ticks, timestep_ticks)
                stack = append_or_update(stack, note, t)
            elif line[2] == 'End_track': 
                state = 1
                stack = {}

# Encoding of note vectors into ascii strings. One timestep is represented as a
# single ascii string; timesteps are separated with a single space. Timesteps
# where no notes are being played are represented as spaces. 
res = ''
for i, vec in enumerate(note_vecs):
    if vec == []: 
        res += ' '
    else:
        vec = sorted(vec)
        res += encode_seq(vec)
    res += ' '

with open(outfile, 'w') as f:
    f.write(res)
