import mido
import time
from mido import MidiFile
import rtmidi
from mido.ports import MultiPort
from midiutil.MidiFile import MIDIFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt


####################################
#####     GLOBAL VARIABLES     #####
####################################

# midi_infile = 'midi_files/Payphone - Maroon 5 ft. Wiz Khalifa [MIDICollection.net].mid'
# midi_infile = 'mid/hisaishi_summer.mid'
# midi_infile = 'midi_files/Someone Like You - Adele [MIDICollection.net].mid'
# # midi_infile = 'midi_files/Rollin In The Deep - Adele [MIDICollection.net].mid'
midi_infile = 'midi_files/River Flows In You - Yiruma [MIDICollection.net].mid'
# midi_infile = 'midi_files/Whistle - Flo Rida [MIDICollection.net].mid'
# midi_infile = 'midi_files/Fly - Ludovico Einaudi [MIDICollection.net].mid'
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

    music_dict = {'id': [], 'notes': [], 'time': [], 'velocity': []}
    id_dict = {}

    mid = MidiFile(midi_file)
    count = 0
    id_count = 0
    for i, track in enumerate(mid.tracks):
        # print('Track {}: {}'.format(i, track.name))
        for message in track:
            # print(message)
            # Filter for note_on and channel == 0
            if message.type == 'note_on':
                if message.channel == 0:
                    count += 1
                    music_dict['id'].append(count)
                    music_dict['notes'].append(message.note)
                    music_dict['time'].append(int(str(message.time)[:2]))
                    music_dict['velocity'].append(message.velocity)

                    if message.note not in id_dict.keys():
                        id_dict[message.note] = id_count
                        id_count += 1


    return music_dict, id_dict


###########################################
#####     CREATE ADJACENCY MATRIX     #####
###########################################

def get_edge_lst(music_dict):
    unweighted_edge_lst = []
    for idx, msg in enumerate(music_dict['notes']):
        try:
            unweighted_edge_lst.append((music_dict['notes'][idx], music_dict['notes'][idx+1]))
        except:
            pass

    weighted_edge_dict = {}
    for ele in unweighted_edge_lst:
        if ele not in weighted_edge_dict:
            weighted_edge_dict[ele] = 1
        else:
            weighted_edge_dict[ele] += 1


    return unweighted_edge_lst, weighted_edge_dict

def write_adjacency_id_matrix(id_dict, weighted_edge_dict):

    adj_mtx = np.zeros((len(id_dict), len(id_dict)))
    id_mtx = np.zeros((len(id_dict), len(id_dict)))
    # print(weighted_edge_dict)
    for idx, key in enumerate(weighted_edge_dict.keys()):
        adj_mtx[id_dict[key[0]], id_dict[key[1]]] = weighted_edge_dict[key]
        id_mtx[id_dict[key[0]], id_dict[key[1]]] = key[0]

    note_id_lst = np.true_divide(id_mtx.sum(1),(id_mtx!=0).sum(1))
    return adj_mtx, id_mtx, note_id_lst

def get_adjacency_matrix_probabilities(adj_mtx, id_dict):
    adj_mtx_probs = adj_mtx/adj_mtx.sum(axis=0)
    return adj_mtx_probs

def get_random_choice(id_dict, music_dict, id_mtx, adj_mtx_probs, emission_mtx, emission_id_mtx):
    # id_seq = []
    # note_seq = []
    # # init_note = random.sample(list(id_dict.values()), 1)[0]
    #
    # # generate using Markov Model
    # init_note = 0
    # id_seq.append(init_note)
    # while len(id_seq) < len(music_dict['notes']):
    #     random_choice = np.random.choice(id_mtx[:, id_seq[-1]], p=adj_mtx_probs[:, id_seq[-1]])
    #     choice_id = id_dict[int(random_choice)]
    #     id_seq.append(choice_id)
    # # print(id_dict)
    # for ele in id_seq:
    #     ids = list(id_dict.keys())
    #     note_seq.append(ids[ele])


    # generate using Hidden Markov Model

    id_seq = []
    int_seq = []
    state_seq = []


    firststate = random.sample(list(id_dict.values()), 1)[0]
    id_seq.append(int(firststate))

    probabilities = emission_mtx[firststate,:]
    # print('probabilities', probabilities)
    # print(emission_mtx)
    # print(emission_id_mtx)
    firstint = np.random.choice(emission_id_mtx[firststate,:], p=emission_mtx[firststate,:])
    firststate = state = np.random.choice(id_mtx[:, firststate], p=adj_mtx_probs[:, firststate])
    # print('first_int', firstint)
    int_seq.append(int(firstint))
    state_seq.append(int(firststate))

    while len(state_seq)<len(music_dict['notes']):
        prevstate = id_seq[-1]
        state = np.random.choice(id_mtx[:, prevstate], p=adj_mtx_probs[:, prevstate])
        choice_id = id_dict[int(state)]
        interval = np.random.choice(emission_id_mtx[choice_id,:], p=emission_mtx[choice_id,:])
        state_seq.append(int(state))
        id_seq.append(int(choice_id))
        int_seq.append(int(interval))

    # return note_seq
    # print(id_seq)
    # print(int_seq)
    # print(state_seq)
    return state_seq, int_seq


##########################################
#####     CREATE EMISSION MATRIX     #####
##########################################


def write_emission_matrix(music_df, note_id_lst):

    chord_intervals = [0]
    time_lst = music_df['time']
    int_lst = music_df['intervals']
    for idx, time in enumerate(time_lst[:-1]):
        if time_lst[idx+1]==0:
            chord_intervals.append(int_lst[idx])
            # elif time_lst[idx-1]==0:
            #     chord_intervals.append(int_lst[idx])
        else:
            chord_intervals.append(0)
     # chord_intervals.append(0)

    for idx, time in enumerate(time_lst):
        print(time_lst[idx], chord_intervals[idx], int_lst[idx])
    print('chord intervals', chord_intervals)
    print('int_lst', list(int_lst))
    print('time_lst', list(time_lst))

    music_df['intervals'] = chord_intervals


    note_sums = []
    for note in note_id_lst:
        note = int(note)
        note_sums.append(music_df.iloc[note].sum())
    # print(len(note_sums))
    # print(len(music_df['notes'].value_counts()))

    emission_mtx = np.zeros((len(music_df['notes'].value_counts()), len(music_df['intervals'].value_counts())))
    emission_id_mtx = emission_mtx.copy()

    for idx, sum in enumerate(note_sums):
        emission_mtx[idx,:] = sum

    interval_id_lst = set(music_df['intervals'].unique())
    for note_idx, note in enumerate(note_id_lst):
        temp_df = music_df[music_df['notes']==note]
        # print(temp_df)
        num_notes = temp_df.shape[0]
        temp_idx_ct = 0
        for int_idx, interval in enumerate(interval_id_lst):
            # print(interval)
            int_df = temp_df[temp_df['intervals']==interval]
            # print(int_df)
            # print(int_df.shape[0])
            # print(note_sums[note_idx])
            num_ints = int_df.shape[0]
            temp_idx_ct += num_ints

            emission_mtx[note_idx,int_idx] = num_ints/num_notes
            emission_id_mtx[note_idx,int_idx] = interval

            # if len(temp_df['intervals']>0):
            #     print(temp_df['intervals'])
            # print(interval, note)

    # emission_mtx_probs = emission_mtx/adj_mtx.sum(axis=1)

    # print(emission_id_mtx)

    return emission_mtx, emission_id_mtx

def write_emission_time_mtx():

    time_mtx = np.zeros((len(music_df['notes'].value_counts()), len(music_df['time'].value_counts())))
    time_id_mtx = time_mtx.copy()


#######################################
#####     WRITE TO DATA FRAME     #####
#######################################

def create_dataframe(music_dict):

    train_df = pd.DataFrame.from_dict(music_dict)

    # print(train_df['notes'].value_counts())

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

def write_midi_file(midi_file, music_dict, note_seq, int_seq):

    # Create MIDI object
    mf = MIDIFile(1)     # 1 track
    track = 0   # the only track

    time = 0  # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 240)

    # add some notes
    channel = 0
    volume = 100

    duration = 2
    temp_time = 0
    speed = .05

    for idx, msg in enumerate(music_dict['notes']):

        # pitch = music_dict['notes'][idx]
        pitch = note_seq[idx]
        accompaniment = int_seq[idx]+pitch

        # print(music_dict['time'][idx])

# uses approximately the original song's time
        if music_dict['time'][idx] < 99:
            # time = idx + music_dict['time'][idx]
            time = temp_time + music_dict['time'][idx]*speed
            temp_time += music_dict['time'][idx]*speed

        else:
            time = temp_time + 16*speed
            temp_time += time

        # time = idx




        mf.addNote(track, channel, pitch, time, duration, volume)
        mf.addNote(track, channel, accompaniment, time, duration, volume)

    # write to disk
    with open(midi_file, 'wb') as outf:
        mf.writeFile(outf)


##################################
#####     CROSS VALIDATE     #####
##################################

def cross_validate(y_test, y_preds):

    random_notes = [random.randrange(74, 86, 1) for _ in range(len(y_test))] # random notes (chosen from notes that appear in the song)
    rms_random = sqrt(mean_squared_error(y_test, random_notes))

    median_notes = [74]*len(y_test)
    rms_median = sqrt(mean_squared_error(y_test, median_notes))

    rms_preds = sqrt(mean_squared_error(y_test, y_preds))

    root_lst = [0]*len(y_test)

    normalized_random_notes = abs(random_notes - y_test)
    normalized_median_notes = abs(median_notes - y_test)
    normalized_pred_notes = abs(y_preds - y_test)

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

    corrected_preds = []
    for note in normalized_pred_notes:
        if note == root:
            corrected_preds.append(0)
        elif note in major:
            corrected_preds.append(1)
        elif note in minor:
            corrected_preds.append(2)
        else:
            corrected_preds.append(3)

    rms_corrected_random = sqrt(mean_squared_error(root_lst, corrected_random))
    rms_corrected_median = sqrt(mean_squared_error(root_lst, corrected_median))
    rms_corrected_preds = sqrt(mean_squared_error(root_lst, corrected_preds))

    print('Random: ', rms_random)
    print('Corrected random: ', rms_corrected_random)
    print('Median: ', rms_median)
    print('Corrected median: ', rms_corrected_median)
    print('HMM Predictions: ', rms_preds)
    print('HMM Corrected Predictions: ', rms_corrected_preds)

    return None

#############################
#####     PLAY SONG     #####
#############################

def play_song(midi_file):

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print('Available ports: ', available_ports)

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
    print('Opening port: ', port)

    for msg in MidiFile(midi_file).play():
        port.send(msg)

    return None


##################################
#####     PLOT SONG DATA     #####
##################################

def get_intervals_lst(music_df):

    interval_lst = []
    note_lst = list(music_df['notes'])
    for idx, ele in enumerate(note_lst):
        interval_lst.append(abs(note_lst[idx] - note_lst[idx - 1]))
        # if idx>0:
        #     interval_lst.append(abs(note_lst[idx]-note_lst[idx-1]))
        # else:
        #     interval_lst.append(0)
    music_df['intervals'] = interval_lst

    return music_df

def plot_song_data(music_df):

    # interval_lst = []
    # note_lst = list(music_df['notes'])
    # for idx, ele in enumerate(note_lst):
    #     try:
    #         interval_lst.append(abs(note_lst[idx]-note_lst[idx-1]))
    #     except:
    #         interval_lst.append(0)
    # music_df['intervals'] = interval_lst

    # print('Total number of notes: {}'.format(len(music_df['id'])))
    # print('Total number of unique notes', len(music_df['notes'].unique()))
    # print('Num notes by name:', music_df['notes'].value_counts())
    # print('Interval types', music_df['intervals'])
    # print('Num unique intervals', len(music_df['intervals'].unique()))
    # print('Num intervals by name:', music_df['intervals'].value_counts())

    # Plot notes
    note_df = music_df.groupby('notes').count()
    note_df = note_df.reset_index()

    ax = sns.barplot(x=note_df['notes'], y=note_df['id'])
    ax.set_title('Note Distribution')
    ax.set(xlabel='Midi Note Numbers', ylabel='Frequency')
    plt.show()

    # Plot intervals
    interval_df = music_df.groupby('intervals').count()
    interval_df = interval_df.reset_index()

    ax = sns.barplot(x=interval_df['intervals'], y=interval_df['id'])
    ax.set_title('Interval Distribution')
    ax.set(xlabel='Intervals', ylabel='Frequency')
    plt.show()


    return None


########################
#####     MAIN     #####
########################

# load_midi_file(midi_infile)
music_dict, id_dict = parse_midi_file(midi_infile)
unweighted_edge_lst, weighted_edge_dict = get_edge_lst(music_dict)
music_df = create_dataframe(music_dict)
music_df = get_intervals_lst(music_df)
adj_mtx, id_mtx, note_id_lst = write_adjacency_id_matrix(id_dict, weighted_edge_dict)
adj_mtx_probs = get_adjacency_matrix_probabilities(adj_mtx, id_dict)
emission_mtx, emission_id_mtx = write_emission_matrix(music_df, note_id_lst)
note_seq, int_seq = get_random_choice(id_dict, music_dict, id_mtx, adj_mtx_probs, emission_mtx, emission_id_mtx)
# music_dict_pred, id_dict_pred = parse_midi_file(midi_outfile)
# music_df_pred = create_dataframe(music_dict_pred)
# predictions = music_df_pred['notes']

# X_train, X_test, y_train, y_test = split_data(music_df)
# cross_validate(music_df['notes'], note_seq)
write_midi_file(midi_outfile, music_dict, note_seq, int_seq)
# play_song(midi_infile)
play_song(midi_outfile)
# plot_song_data(music_df)


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