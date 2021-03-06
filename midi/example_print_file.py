"""
This is an example that uses the MidiToText eventhandler. When an 
event is triggered on it, it prints the event to the console.

It gets the events from the MidiInFile.

So it prints all the events from the infile to the console. great for 
debugging :-s
"""


# get data
test_file = 'test/midifiles/002.mid'

# do parsing
from midi.MidiInFile import MidiInFile
from midi.MidiToText import MidiToText # the event handler
midiIn = MidiInFile(MidiToText(), test_file)
midiIn.read()
