import mido
import time
from mido import MidiFile
import rtmidi
from mido.ports import MultiPort
from midiutil.MidiFile import MIDIFile


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

    mid = MidiFile(midi_file)
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for message in track:
            # Filter for note_on and channel == 0
            if message.type == 'note_on':
                if message.channel == 0:
                    print(message)


######################################
#####     WRITE TO MIDI FILE     #####
######################################

def write_midi_file(midi_file):

    # Create MIDI object
    mf = MIDIFile(1)     # 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)

    # add some notes
    channel = 0
    volume = 100

    pitch = 60           # C4 (middle C)
    time = 0             # start on beat 0
    duration = 1         # 1 beat long
    mf.addNote(track, channel, pitch, time, duration, volume)

    pitch = 64           # E4
    time = 2             # start on beat 2
    duration = 1         # 1 beat long
    mf.addNote(track, channel, pitch, time, duration, volume)

    pitch = 67           # G4
    time = 4             # start on beat 4
    duration = 1         # 1 beat long
    mf.addNote(track, channel, pitch, time, duration, volume)

    # write it to disk
    with open(midi_file, 'wb') as outf:
        mf.writeFile(outf)

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

load_midi_file(midi_infile)
parse_midi_file(midi_infile)
write_midi_file(midi_outfile)
play_song(midi_outfile)