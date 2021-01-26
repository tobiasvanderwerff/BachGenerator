import argparse

import py_midicsv as pm


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile')
parser.add_argument('-o', '--outfile')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile

# Load the MIDI file and parse it into CSV format
csv_string = pm.midi_to_csv(infile)

with open(outfile, "w") as f:
    f.write("".join(csv_string))
