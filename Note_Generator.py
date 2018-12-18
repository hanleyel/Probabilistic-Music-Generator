import mido
from mido import MidiFile
import rtmidi
from midiutil.MidiFile import MIDIFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import LabelEncoder
import csv

####################################
#####     GLOBAL VARIABLES     #####
####################################

configfiles = glob.glob("midi_files/**/*.mid", recursive=True)

midi_infile = configfiles[2]
midi_outfile = 'test-output.mid'

###################################
#####     PARSE MIDI FILE     #####
###################################


def write_music_dataframe(midi_file):

    music_df = pd.DataFrame(columns=['note', 'time', 'velocity'])
    mid = MidiFile(midi_file)
    count = 0
    for i, track in enumerate(mid.tracks):
        for message in track:
            if message.type == 'note_on' and message.channel == 0:
                if count == 0:
                    prev_note = message.note # set prev_note to first note of the song
                music_df.loc[count] = [message.note, int(str(message.time)[:2]), message.velocity]
                prev_note = message.note
                count += 1

    prog_intervals = []
    chord_intervals = []
    time_lst = music_df['time']
    note_lst = music_df['note']
    for idx, time in enumerate(time_lst[:-1]):
        prog_interval = note_lst[idx+1] - note_lst[idx]
        prog_intervals.append(prog_interval)
        if time_lst[idx+1]==0:
            chord_intervals.append(prog_interval)
        else:
            chord_intervals.append(0)

    prog_intervals.append(0)
    chord_intervals.append(0)

    music_df['prog_interval'] = prog_intervals
    music_df['chord_interval'] = chord_intervals
    music_df['next_note'] = music_df['note']+music_df['prog_interval']

    le = LabelEncoder()
    le.fit(music_df['note'])
    music_df['id'] = le.transform(music_df['note'])
    # print(music_df)
    return music_df

music_df = write_music_dataframe(midi_infile)


##############################################################
#####     CALCULATE TRANSITION AND EMISSION MATRICES     #####
##############################################################

def write_matrices(music_df):

    note_id_lst = set(music_df['note'].unique())
    interval_id_lst = set(music_df['chord_interval'].unique())
    time_id_lst = set(music_df['time'].unique())
    volume_id_lst = set(music_df['velocity'].unique())

    transition_mtx = np.zeros((len(music_df['note'].value_counts()), len(music_df['note'].value_counts())))
    transition_id_mtx = transition_mtx.copy()

    emission_mtx = np.zeros((len(music_df['note'].value_counts()), len(music_df['chord_interval'].value_counts())))
    emission_id_mtx = emission_mtx.copy()

    time_mtx = np.zeros((len(music_df['note'].value_counts()), len(music_df['time'].value_counts())))
    time_id_mtx = time_mtx.copy()

    volume_mtx = np.zeros((len(music_df['note'].value_counts()), len(music_df['velocity'].value_counts())))
    volume_id_mtx = volume_mtx.copy()


    for note_idx, note in enumerate(note_id_lst):
        temp_df = music_df[music_df['note']==note]
        num_notes = temp_df.shape[0]

        for note_idx_2, note_2 in enumerate(note_id_lst):
            note_df = temp_df[temp_df['next_note']==note_2]
            num_notes_2 = note_df.shape[0]
            transition_mtx[note_idx,note_idx_2] = num_notes_2/num_notes
            transition_id_mtx[note_idx,note_idx_2] = note_2

        for int_idx, interval in enumerate(interval_id_lst):
            int_df = temp_df[temp_df['chord_interval']==interval]
            num_ints = int_df.shape[0]
            emission_mtx[note_idx,int_idx] = num_ints/num_notes
            emission_id_mtx[note_idx,int_idx] = interval

        for time_idx, time in enumerate(time_id_lst):
            time_df = temp_df[temp_df['time']==time]
            num_times = time_df.shape[0]
            time_mtx[note_idx,time_idx] = num_times/num_notes
            time_id_mtx[note_idx,time_idx] = time


        for velocity_idx, velocity in enumerate(volume_id_lst):
            volume_df = temp_df[temp_df['velocity']==velocity]
            num_velocities = volume_df.shape[0]
            volume_mtx[note_idx,velocity_idx] = num_velocities/num_notes
            volume_id_mtx[note_idx,velocity_idx] = velocity

    # print(transition_mtx)
    # print(transition_id_mtx)
    # print(emission_mtx)
    # print(emission_id_mtx)
    # print(volume_mtx)
    # print(volume_id_mtx)

    return transition_mtx, transition_id_mtx, emission_mtx, emission_id_mtx, time_mtx, time_id_mtx, volume_mtx, volume_id_mtx

transition_mtx, transition_id_mtx, emission_mtx, emission_id_mtx, time_mtx, time_id_mtx, volume_mtx, volume_id_mtx = write_matrices(music_df)


# def parse_midi_file(midi_file):
#
#     music_dict = {'id': [], 'notes': [], 'time': [], 'velocity': []}
#     id_dict = {}
#     mid = MidiFile(midi_file)
#     count = 0
#     id_count = 0
#     for i, track in enumerate(mid.tracks):
#         # print('Track {}: {}'.format(i, track.name))
#         for message in track:
#             # print(message)
#             if message.type == 'note_on':
#                 if message.channel == 0:
#                     count += 1
#                     music_dict['id'].append(count)
#                     music_dict['notes'].append(message.note)
#                     music_dict['time'].append(int(str(message.time)[:2]))
#                     music_dict['velocity'].append(message.velocity)
#                     if message.note not in id_dict.keys():
#                         id_dict[message.note] = id_count
#                         id_count += 1
#
#     return music_dict, id_dict


###########################################
#####     CREATE ADJACENCY MATRIX     #####
###########################################

# def get_edge_lst(music_dict):
#
#     unweighted_edge_lst = []
#     for idx, msg in enumerate(music_dict['notes']):
#         try:
#             unweighted_edge_lst.append((music_dict['notes'][idx], music_dict['notes'][idx+1]))
#         except:
#             pass
#
#     weighted_edge_dict = {}
#     for ele in unweighted_edge_lst:
#         if ele not in weighted_edge_dict:
#             weighted_edge_dict[ele] = 1
#         else:
#             weighted_edge_dict[ele] += 1
#
#     return unweighted_edge_lst, weighted_edge_dict
#
#
# def write_adjacency_id_matrix(id_dict, weighted_edge_dict):
#
#     adj_mtx = np.zeros((len(id_dict), len(id_dict)))
#     id_mtx = np.zeros((len(id_dict), len(id_dict)))
#     # print(weighted_edge_dict)
#     for idx, key in enumerate(weighted_edge_dict.keys()):
#         adj_mtx[id_dict[key[0]], id_dict[key[1]]] = weighted_edge_dict[key]
#         id_mtx[id_dict[key[0]], id_dict[key[1]]] = key[0]
#
#     note_id_lst = np.true_divide(id_mtx.sum(1),(id_mtx!=0).sum(1))
#     return adj_mtx, id_mtx, note_id_lst
#
#
# def get_adjacency_matrix_probabilities(adj_mtx, id_dict):
#     adj_mtx_probs = adj_mtx/adj_mtx.sum(axis=0)
#     return adj_mtx_probs


def get_random_choice(emission_mtx, emission_id_mtx, time_mtx, time_id_mtx,
                      volume_mtx, volume_id_mtx, transition_mtx, transition_id_mtx, music_df):

    id_dict = dict(zip(music_df['note'], music_df['id']))

    id_seq = []
    int_seq = []
    time_seq = []
    vol_seq = []
    state_seq = []

    # firststate = random.sample(list(id_dict.values()), 1)[0]
    firststate = random.sample(list(music_df['id'].unique()), 1)[0]
    id_seq.append(firststate)

    # probabilities = emission_mtx[firststate,:]
    firstint = np.random.choice(emission_id_mtx[firststate,:], p=emission_mtx[firststate,:])
    firsttime = np.random.choice(time_id_mtx[firststate,:], p=time_mtx[firststate,:])
    firstvol = np.random.choice(volume_id_mtx[firststate, :], p=volume_mtx[firststate, :])
    firststate = np.random.choice(transition_id_mtx[firststate,:], p=transition_mtx[firststate,:])
    int_seq.append(int(firstint))
    time_seq.append(int(firsttime))
    vol_seq.append(int(firstvol))
    state_seq.append(int(firststate))

    while len(state_seq)<len(music_df['note']):
        prevstate = id_seq[-1]
        state = np.random.choice(transition_id_mtx[prevstate,:], p=transition_mtx[prevstate,:])
        choice_id = id_dict[state]
        interval = np.random.choice(emission_id_mtx[choice_id,:], p=emission_mtx[choice_id,:])
        time_state = np.random.choice(time_id_mtx[choice_id,:], p=time_mtx[choice_id,:])
        vol_state = np.random.choice(volume_id_mtx[choice_id, :], p=volume_mtx[choice_id, :])
        state_seq.append(int(state))
        id_seq.append(int(choice_id))
        int_seq.append(int(interval))
        time_seq.append(int(time_state))
        vol_seq.append(int(vol_state))

    return state_seq, int_seq, time_seq, vol_seq




#######################################
#####     WRITE TO DATA FRAME     #####
#######################################

# def create_dataframe(music_dict):
#
#     train_df = pd.DataFrame.from_dict(music_dict)
#
#     return train_df


###############################################
#####     SPLIT DATA (TRAIN AND TEST)     #####
###############################################

def split_data(music_df):

    X_train, X_test, y_train, y_test = train_test_split(music_df['id'], music_df['notes'], random_state=0)

    return X_train, X_test, y_train, y_test


######################################
#####     WRITE TO MIDI FILE     #####
######################################

def write_midi_file(midi_file, note_seq, int_seq, time_seq, vol_seq):

    # Create MIDI object
    mf = MIDIFile(1)     # 1 track
    track = 0   # the only track

    time = 0  # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 240)

    # add some notes
    channel = 0
    volume = 100

    duration = 10
    temp_time = 0
    speed = .025

    for idx, msg in enumerate(note_seq):

        # pitch = music_dict['notes'][idx]
        pitch = note_seq[idx]
        volume = vol_seq[idx]
        # print(volume)

# uses the HMM generated time
        if time_seq[idx] < 99:
            # time = idx + music_dict['time'][idx]
            time = temp_time + time_seq[idx]*speed
            temp_time += time_seq[idx]*speed

        else:
            time = temp_time + 16*speed
            temp_time += time

        try:
            nexttime = time_seq[idx+1]*speed
        except:
            nexttime = time_seq[idx]*speed
        if nexttime <= 1:
            nexttime = 1
        # duration = nexttime

        mf.addNote(track, channel, pitch, time, duration, volume)

        try:
            if int_seq[idx] > 0:
                accompaniment = int_seq[idx] + pitch
                try:
                    if int_seq[idx] == 1 or int_seq[idx] == 2:
                        mf.addNote(track, channel, accompaniment, time, duration, volume)
                except:
                    mf.addNote(track, channel, accompaniment, time+1, duration, volume)
                # print(int_seq[idx])
        except:
            pass

    # write to disk
    with open(midi_file, 'wb') as outf:
        mf.writeFile(outf)


##################################
#####     CROSS VALIDATE     #####
##################################

def cross_validate(y_test, y_preds):

    note_min = min(y_test)
    note_max = max(y_test)

    random_notes = [random.randrange(note_min, note_max, 1) for _ in range(len(y_test))] # random notes (chosen from notes that appear in the song)
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

    return rms_random, rms_corrected_random, rms_median, rms_corrected_median, rms_preds, rms_corrected_preds

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

    # note_on = [0x90, 60, 112] # channel 1, middle C, velocity 112
    # note_off = [0x80, 60, 0]
    # midiout.send_message(note_on)
    # time.sleep(0.5)
    # midiout.send_message(note_off)

    del midiout

    port = mido.open_output()
    print('Opening port: ', port)

    for msg in MidiFile(midi_file).play():
        port.send(msg)

    return None


##################################
#####     PLOT SONG DATA     #####
##################################

# def get_intervals_lst(music_df):
#
#     interval_lst = []
#     note_lst = list(music_df['notes'])
#     for idx, ele in enumerate(note_lst):
#         interval_lst.append(abs(note_lst[idx] - note_lst[idx - 1]))
#         # if idx>0:
#         #     interval_lst.append(abs(note_lst[idx]-note_lst[idx-1]))
#         # else:
#         #     interval_lst.append(0)
#     music_df['intervals'] = interval_lst
#
#     return music_df

def plot_song_data(music_df):

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

# music_dict, id_dict = parse_midi_file(midi_infile)
# print('Music dict', music_dict)
# print('id dict', id_dict)
# unweighted_edge_lst, weighted_edge_dict = get_edge_lst(music_dict)
# print('unweighted edge', unweighted_edge_lst)
# print('weighted edge', weighted_edge_dict)
# music_df = create_dataframe(music_dict)
# print('music df', music_df)
# music_df = get_intervals_lst(music_df)
# adj_mtx, id_mtx, note_id_lst = write_adjacency_id_matrix(id_dict, weighted_edge_dict)
# adj_mtx_probs = get_adjacency_matrix_probabilities(adj_mtx, id_dict)
# emission_mtx, emission_id_mtx, time_mtx, time_id_mtx, volume_mtx, volume_id_mtx = write_emission_matrix(music_df, note_id_lst)
# # time_mtx, time_id_mtx = write_emission_time_mtx(music_df, note_id_lst)
note_seq, int_seq, time_seq, vol_seq = get_random_choice(emission_mtx, emission_id_mtx,
                                                time_mtx, time_id_mtx, volume_mtx, volume_id_mtx, transition_mtx, transition_id_mtx, music_df)
#
#
# # X_train, X_test, y_train, y_test = split_data(music_df)
#
write_midi_file(midi_outfile, note_seq, int_seq, time_seq, vol_seq)
# # music_dict_pred, id_dict_pred = parse_midi_file(midi_outfile)
# # music_df_pred = create_dataframe(music_dict_pred)
# # predictions = music_df_pred['notes']
# # rms_random, rms_corrected_random, rms_median, rms_corrected_median, rms_preds, rms_corrected_preds = cross_validate(music_df['notes'], note_seq)
# # play_song(midi_infile)
play_song(midi_outfile)
# # plot_song_data(music_df)

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