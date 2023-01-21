from music21 import converter, instrument, note, chord
import os
from sys import platform
import numpy as np
from imageio import imwrite
from pathlib import Path
import argparse

def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))
                
        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element.notes:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

def midi2image(midi_path, input_reps):
    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = get_notes(notes_to_parse)
                i+=1
            else:
                data[instrument_i.partName] = get_notes(notes_to_parse)

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0".format(i)] = get_notes(notes_to_parse)

    resolution = 0.25

    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
        upperBoundNote = 127
        lowerBoundNote = 21
        maxSongLength = 100

        index = 0
        prev_index = 0
        repetitions = 0
        while repetitions < input_reps:
            if prev_index >= len(values["pitch"]):
                break

            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))

            pitchs = values["pitch"]
            durs = values["dur"]
            starts = values["start"]

            for i in range(prev_index,len(pitchs)):
                pitch = pitchs[i]

                dur = int(durs[i]/resolution)
                start = int(starts[i]/resolution)

                if dur+start - index*maxSongLength < maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0:
                            matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255
                else:
                    prev_index = i
                    break
            
            par = os.path.dirname(Path(__file__).parent)
            fname = ""
            if platform == "linux" or platform == "linux2":
                # linux
                fname = str(midi_path).split("/")[-1].replace(".mid",f"_{instrument_name}_{index}.png")
            elif platform == "darwin":
                # OS X
                fname = str(midi_path).split("/")[-1].replace(".mid",f"_{instrument_name}_{index}.png")
            elif platform == "win32":
                # Windows...
                fname = str(midi_path).split("\\")[-1].replace(".mid",f"_{instrument_name}_{index}.png")

            save_path = os.path.join(par, "TrainingImages", "0", fname)
            print("Saved at: {}".format(save_path))
            imwrite(save_path, matrix)
            index += 1
            repetitions+=1

if __name__ == "__main__": 
    parent = os.path.dirname(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=Path(os.path.join(parent, "MidiTrain", "jazz_1.mp3.mid")))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("-d", '--isdir', action='store_true', dest='isdir')
    parser.set_defaults(isdir=True)
    opt = parser.parse_args()

    if not opt.isdir:
        print("Just one file selected")
        midi2image(opt.path, opt.repetitions)
    else:
        print("directory selected")
        for root, dirs, files in os.walk(opt.path):
            for file in files:
                if file.endswith(".mid"):
                    pt = os.path.join(opt.path, file)
                    midi2image(pt, opt.repetitions)
