# -*- coding: ISO-8859-1 -*-

from midi.MidiOutStream import MidiOutStream


class MidiReadList(MidiOutStream):
    alltime = 0
    ontime = 0
    offtime = 0
    note_count = 0
    time_freq = [[0 for col in range(128)] for row in range(128)]
    note_freq = [0 for row in range(128)]
    vel_freq = [0 for row in range(128)]
    now_note = []
    now_table = [0 for row in range(128)]
    note_list = []
#    time_note = [0 for row in range(128)]

    """
    This class renders a midi file as text. It is mostly used for debugging
    """

    #############################
    # channel events


    def channel_message(self, message_type, channel, data):
        """The default event handler for channel messages"""
        print('message_type:%X, channel:%X, data size:%X' % (message_type, channel, len(data)))

    def note_on(self, channel=0, note=0x40, velocity=0x40):
        #print('note_on  - ch:%02X,  note:%02X,  vel:%02X time:%s' % (channel, note, velocity, self.rel_time()))
        time = self.rel_time()
        for index in self.now_note:
            no, vel = index
            self.now_table[no] += time

        self.note_count += 1
        self.ontime += self.rel_time()
        self.alltime += self.rel_time()
        self.now_note.append([note, velocity])
        self.time_freq[note][velocity] += 1
        self.note_freq[note] += 1
        self.vel_freq[velocity] += 1


    def note_off(self, channel=0, note=0x40, velocity=0x40):
        #print('note_off - ch:%02X,  note:%02X,  vel:%02X time:%s' % (channel, note, velocity, self.rel_time()))
        self.offtime += self.rel_time()
        self.alltime += self.rel_time()
        time = self.rel_time()
        for i in range(len(self.now_note)):
            no, vel = self.now_note[i]
            if no == note:
                if vel <= 15:
                    re_vel = 0
                elif vel <= 25 and vel > 15:
                    re_vel = 1
                elif vel <= 35 and vel > 25:
                    re_vel = 2
                elif vel <= 45 and vel > 35:
                    re_vel = 3
                elif vel <= 55 and vel > 45:
                    re_vel = 4
                elif vel <= 65 and vel > 55:
                    re_vel = 5
                elif vel <= 75 and vel > 65:
                    re_vel = 6
                elif vel <= 85 and vel > 75:
                    re_vel = 7
                elif vel > 85:
                    re_vel = 8
                re_no = no
                if no < 21:
                    re_no = 21
                if no > 100:
                    re_no = 100
                re_no -= 21

                length = self.now_table[no] + time
                if length <= 3:
                    re_length = 0
                elif length <= 7 and length > 3:
                    re_length = 1
                elif length <= 12 and length > 7:
                    re_length = 2
                elif length <= 17 and length > 12:
                    re_length = 3
                elif length <= 25 and length > 17:
                    re_length = 4
                elif length <= 35 and length > 25:
                    re_length = 5
                elif length > 35:
                    re_length = 6


                self.note_list.append([self.alltime - length, re_no, re_vel, re_length])
#                self.time_note[length] += 1
                del self.now_note[i]
                self.now_table[no] = 0
                break
        for index in self.now_note:
            no, vel = index
            self.now_table[no] += time

    def aftertouch(self, channel=0, note=0x40, velocity=0x40):
        print('aftertouch', channel, note, velocity)

    def continuous_controller(self, channel, controller, value):
        print('controller - ch: %02X, cont: #%02X, value: %02X' % (channel, controller, value))

    def patch_change(self, channel, patch):
        print('patch_change - ch:%02X, patch:%02X' % (channel, patch))

    def channel_pressure(self, channel, pressure):
        print('channel_pressure', channel, pressure)

    def pitch_bend(self, channel, value):
        print('pitch_bend ch:%s, value:%s' % (channel, value))

    #####################
    ## Common events


    def system_exclusive(self, data):
        print('system_exclusive - data size: %s' % len(data))

    def song_position_pointer(self, value):
        print('song_position_pointer: %s' % value)

    def song_select(self, songNumber):
        print('song_select: %s' % songNumber)

    def tuning_request(self):
        print('tuning_request')

    def midi_time_code(self, msg_type, values):
        print('midi_time_code - msg_type: %s, values: %s' % (msg_type, values))

    #########################
    # header does not really belong here. But anyhoo!!!

    def header(self, format=0, nTracks=1, division=96):
        print('format: %s, nTracks: %s, division: %s' % (format, nTracks, division))
        print('----------------------------------')
        print('')

    def eof(self):
        print('End of file')

    def start_of_track(self, n_track=0):
        print('Start - track #%s' % n_track)

    def end_of_track(self):
        print('End of track')
        print(self.alltime)
 #       print(self.ontime)
 #       print(self.offtime)
        print('')

    ###############
    # sysex event

    def sysex_event(self, data):
        print('sysex_event - datasize: %X' % len(data))

    #####################
    ## meta events

    def meta_event(self, meta_type, data):
        print('undefined_meta_event:', meta_type, len(data))

    def sequence_number(self, value):
        print('sequence_number', value)

    def text(self, text):
        print('text', text)

    def copyright(self, text):
        print('copyright', text)

    def sequence_name(self, text):
        print('sequence_name:', text)

    def instrument_name(self, text):
        print('instrument_name:', text)

    def lyric(self, text):
        print('lyric', text)

    def marker(self, text):
        print('marker', text)

    def cuepoint(self, text):
        print('cuepoint', text)

    def midi_ch_prefix(self, channel):
        print('midi_ch_prefix', channel)

    def midi_port(self, value):
        print('midi_port:', value)

    def tempo(self, value):
        print('tempo:', value)

    def smtp_offset(self, hour, minute, second, frame, framePart):
        print('smtp_offset', hour, minute, second, frame, framePart)

    def time_signature(self, nn, dd, cc, bb):
        print('time_signature:', nn, dd, cc, bb)

    def key_signature(self, sf, mi):
        print('key_signature', sf, mi)

    def sequencer_specific(self, data):
        print('sequencer_specific', len(data))


if __name__ == '__main__':
    # get data
    test_file = 'test/midifiles/minimal.mid'
    f = open(test_file, 'rb')

    # do parsing
    from MidiInFile import MidiInFile

    midiIn = MidiInFile(MidiToText(), f)
    midiIn.read()
    f.close()
