import mido
import time
from mido import MidiFile
import rtmidi
from mido.ports import MultiPort
from midiutil.MidiFile import MIDIFile
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error
from math import sqrt


####################################
#####     GLOBAL VARIABLES     #####
####################################

midi_infile = 'mid/hisaishi_summer.mid'
midi_outfile = 'test-output.mid'


##############################
#####     MIDO STUFF     #####
##############################

# msg.type (note on or off)
# msg.note
# msg.velocity


##################################
#####     LOAD MIDI FILE     #####
##################################

def load_midi_file(midi_file):

    mid = MidiFile(midi_file)

    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for message in track:
            print(message)


###################################
#####     PARSE MIDI FILE     #####
###################################

def parse_midi_file(midi_file):

    music_dict = {'id': [], 'notes':[], 'time':[], 'velocity':[]}


    mid = MidiFile(midi_file)
    count = 0
    for i, track in enumerate(mid.tracks):
        # print('Track {}: {}'.format(i, track.name))
        for message in track:
            # Filter for note_on and channel == 0
            if message.type == 'note_on':
                if message.channel == 0:
                    count += 1
                    music_dict['id'].append(count)
                    music_dict['notes'].append(message.note)
                    music_dict['time'].append(message.time)
                    music_dict['velocity'].append(message.velocity)

    return music_dict


#######################################
#####     WRITE TO DATA FRAME     #####
#######################################

def create_dataframe(music_dict):

    train_df = pd.DataFrame.from_dict(music_dict)

    print(train_df['notes'].value_counts())

    return train_df


###############################################
#####     SPLIT DATA (TRAIN AND TEST)     #####
###############################################

def split_data(music_df):

    X_train, X_test, y_train, y_test = train_test_split(music_df['id'], music_df['notes'], random_state=0)

    return X_train, X_test, y_train, y_test


######################################
#####     WRITE TO MIDI FILE     #####
######################################

def write_midi_file(midi_file, music_dict):

    # Create MIDI object
    mf = MIDIFile(1)     # 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)

    # add some notes
    channel = 0
    volume = 100

    for idx, msg in enumerate(music_dict['notes']):

        pitch = music_dict['notes'][idx]
        time = music_dict['time'][idx]+idx
        duration = 1

        mf.addNote(track, channel, pitch, time, duration, volume)

    # write to disk
    with open(midi_file, 'wb') as outf:
        mf.writeFile(outf)


##################################
#####     CROSS VALIDATE     #####
##################################

def cross_validate(X_test, y_test):

    random_notes = [random.randrange(74, 86, 1) for _ in range(len(y_test))] # random notes (chosen from notes that appear in the song)
    rms_random = sqrt(mean_squared_error(y_test, random_notes))

    median_notes = [74]*len(y_test)
    rms_median = sqrt(mean_squared_error(y_test, median_notes))

    root_lst = [0]*len(y_test)

    normalized_random_notes = abs(random_notes - y_test)
    normalized_median_notes = abs(median_notes - y_test)

    root = 0
    major = [4, 7, 11]
    minor = [3, 10]

    corrected_random = []
    for note in normalized_random_notes:
        if note == root:
            corrected_random.append(0)
        elif note in major:
            corrected_random.append(1)
        elif note in minor:
            corrected_random.append(2)
        else:
            corrected_random.append(3)


    corrected_median = []
    for note in normalized_median_notes:
        if note == root:
            corrected_median.append(0)
        elif note in major:
            corrected_median.append(1)
        elif note in minor:
            corrected_median.append(2)
        else:
            corrected_median.append(3)

    rms_corrected_random = sqrt(mean_squared_error(root_lst, corrected_random))
    rms_corrected_median = sqrt(mean_squared_error(root_lst, corrected_median))

    print('Random: ', rms_random)
    print('Corrected random: ', rms_corrected_random)
    print('Median: ', rms_median)
    print('Corrected median: ', rms_corrected_median)

    return None

#############################
#####     PLAY SONG     #####
#############################

def play_song(midi_file):

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)

    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    note_on = [0x90, 60, 112] # channel 1, middle C, velocity 112
    note_off = [0x80, 60, 0]
    midiout.send_message(note_on)
    time.sleep(0.5)
    midiout.send_message(note_off)

    del midiout

    port = mido.open_output()
    print(port)

    for msg in MidiFile(midi_file).play():
        port.send(msg)

    return None

########################
#####     MAIN     #####
########################

# load_midi_file(midi_infile)
music_dict = parse_midi_file(midi_infile)
music_df = create_dataframe(music_dict)
X_train, X_test, y_train, y_test = split_data(music_df)
cross_validate(X_test, y_test)
write_midi_file(midi_outfile, music_dict)
# play_song(midi_outfile)


"""
C major 7th

C: 60
E: 64
G: 67
B: 71

C minor 7th
C: 60
Eb: 63
G: 67
Bb: 70

Hierarchy of "correctness" (lower numbers are best)

absolute value of predicted - actual (or modulo?):

exact:
0

penalty: 0

major 7th
4
7
11

penalty: 1 (if major song)

minor 7th
3
10

penalty: 1 (if minor song)

random:
1
2
5
6
8
9

penalty: 2

"""