import mido
import time
from mido import MidiFile
import rtmidi
from mido.ports import MultiPort

mid = MidiFile('mz_331_1.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for message in track:
        print(message)


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

# for msg in MidiFile('mz_331_1.mid'):
#     time.sleep(msg.time)
#     if not msg.is_meta:
#         port.send(msg)