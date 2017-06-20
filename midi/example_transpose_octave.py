from MidiOutFile import MidiOutFile
from MidiInFile import MidiInFile
import numpy as np

"""
This is an example of the smallest possible type 0 midi file, where 
all the midi events are in the same track.
"""


class Transposer(MidiOutFile):
    
    "Transposes all notes by 1 octave"
    
    def _transp(self, ch, note):
        if ch != 9: # not the drums!
            note += 12
            if note > 127:
                note = 127
        return note

    def reduct_velocity(self, ch, velocity):
        if velocity <= 15:
            velocity = 10
        elif velocity <= 25 and velocity > 15:
            velocity = 20
        elif velocity <= 35 and velocity > 25:
            velocity = 30
        elif velocity <= 45 and velocity > 35:
            velocity = 40
        elif velocity <= 55 and velocity > 45:
            velocity = 50
        elif velocity <= 65 and velocity > 55:
            velocity = 60
        elif velocity <= 75 and velocity > 65:
            velocity = 70
        elif velocity <= 85 and velocity > 75:
            velocity = 80
        elif velocity > 85:
            velocity = 90

#        if velocity <= 24:
#            velocity = 16
#        elif velocity <= 40 and velocity > 24:
#            velocity = 32
#        elif velocity <= 56 and velocity > 40:
#            velocity = 48
#        elif velocity <= 72 and velocity > 56:
#            velocity = 64
#        elif velocity <= 88 and velocity > 72:
#            velocity = 80
#        elif velocity > 88:
#            velocity = 96
        return velocity

    def reduct_note(self, ch, note):
        if note < 20:
            note = 20
        if note > 100:
            note = 100

        return note



    def note_on(self, channel=0, note=0x40, velocity=0x40):
        #note = self._transp(channel, note)
        note = self.reduct_note(channel, note)
        velocity = self.reduct_velocity(channel, velocity)
        MidiOutFile.note_on(self, channel, note, velocity)
        
        
    def note_off(self, channel=0, note=0x40, velocity=0x40):
        #note = self._transp(channel, note)
        note = self.reduct_note(channel, note)
        MidiOutFile.note_off(self, channel, note, velocity)


out_file = 'test/midifiles/trans.mid'
midi_out = Transposer(out_file)

#in_file = 'midiout/minimal_type0.mid'
#in_file = 'test/midifiles/Lola.mid'
in_file = 'test/midifiles/401.mid'
midi_in = MidiInFile(midi_out, in_file)
midi_in.read()


