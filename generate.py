import os
import pdb
import copy
from threading import Thread
from socketserver import ThreadingMixIn
from parse_files import *
from datetime import datetime
import utils.sequence_generator as sequence_generator
import utils.network_utils as network_utils
import config.config_mg as nn_config
from midi.MidiInFile import MidiInFile
from midi.MidiReadList import MidiReadList # the event handler
from midi.MidiOutFile import MidiOutFile


recv_file = './test/' + 'veryveryvery.mp3'
recv_file_tmp = './test/' + 'received_music_tmp.mp3'
recv_file_wav = './test/' + 'received_music_wav.wav'
recv_file_mid = './test/' + 'received_music_mid.mid'
send_file = './test/' + 'sending_music.mp3'
send_file_wav = './test/' + 'sending_music_wav.wav'
send_file_mid = './test/' + 'sending_music_mid.mid'
x_file = './test/'

start_time = 00
end_time = 10
finish_time = 30

# Figure out how many frequencies we have in the data
# freq_space_dims = X_train.shape[2]
config = nn_config.get_neural_net_configuration()
model_basename = config['model_basename']
hidden_dims = config['hidden_dimension_size']
num_dims = config['num_dims']
# Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=num_dims,
                                          num_hidden_dimensions=hidden_dims)



#2phase
#lame  mp3 -> wav

sample_freq_str = "{0:.1f}".format(float(44100) / 1000.0)
#        cmd = 'ffmpeg -i {0} -ar 44100 -ac 1 -c:a libmp3lame {1}'.format(recv_file, recv_file_tmp)
cmd = 'ffmpeg -i {0} -ss {1} -t {2} -ac 1 {3}'.format(recv_file, start_time, end_time - start_time, recv_file_tmp)
#cmd = 'ffmpeg -i {0} -ac 1 {3}'.format(recv_file, start_time, end_time - start_time, recv_file_tmp)
#cmd = 'lame -a -m m {0} {1}'.format(recv_file, recv_file_tmp)
#recov 파일 포맷 -> 3gp or mp3
os.system(cmd)

cmd = 'lame --decode {0} {1} --resample {2}'.format(recv_file_tmp, recv_file_wav, sample_freq_str)
os.system(cmd)

cmd = 'waon -i {0} -o {1} -w 3 -n 4096 -s 2048' .format(recv_file_wav, recv_file_mid)
os.system(cmd)
#process wav -> nparray
#xdata 추출
#이건 필히 지울것!!
#        pdb.set_trace()
#convert_wav_files_to_nptensor_service(recv_file_wav, 4410.0, 100, x_file)
#이건 트레이닝을 위한 X파일 생성이니 변경이 필요

#process nparaay -> generation file
#Xdata set이랑 시작시간, 끝시간, 총 길이
#wav 파일 재생 시간 리턴

#model = network_utils.create_lstm_network(num_frequency_dimensions=8820,
#                                          num_hidden_dimensions=1024)
#model_reverse = network_utils.create_lstm_network(num_frequency_dimensions=8820,
#                                          num_hidden_dimensions=1024)

model.load_weights(model_basename)
#model_reverse.load_weights('./Weights_reverse')

# do parsing

midiIn = MidiInFile(MidiReadList(), recv_file_mid)
midiIn.read()

midi_list = []
midi_list = midiIn.parser.dispatch.outstream.note_list

num_examples = 1
max_seq_len = midiIn.parser.dispatch.outstream.alltime

vector_0 = np.zeros(num_dims)

out_shape = (num_examples, max_seq_len, num_dims)

x_data = np.zeros(out_shape)

num_note = 0
num_no_note = 0
ratio_note = 0.0

dim2_note_shape = (80, 80)
dim1_note_shape = (80)
correlation_note = np.ones(dim2_note_shape)
together_note = np.ones(dim2_note_shape)
rest_length = np.zeros(dim1_note_shape)
count_note = np.zeros(dim1_note_shape)

midi_list.sort()
last_time = 0
for i in range(len(midi_list)):
    x_data[0][midi_list[i][0]][midi_list[i][1] + (midi_list[i][2] * 80) + (midi_list[i][3] * 720)] = 1
    count_note[midi_list[i][1]] += 1

for i in range(max_seq_len):
    if np.array_equal(vector_0, x_data[0][i]):
        x_data[0][i][num_dims - 1] = 1
        num_no_note += 1
#pdb.set_trace()
max_count = 0
num_note = max_seq_len - num_no_note
ratio_note = num_note / max_seq_len
for i in range(count_note.shape[0]):
    if count_note[i] > max_count:
        max_count = count_note[i]

#pdb.set_trace()
output1 = sequence_generator.generate_from_seed(model=model,
                                                seedSeq=x_data,
                                               sequence_length = finish_time - end_time,
                                                thresh_hold=0.98,
                                                blank_ratio = 1 - ratio_note,
                                                average_note = max_count / num_note)
#output2 = sequence_generator.generate_from_seed(model=model_reverse,
#                                                seed=X_train_reverse,
#                                                sequence_length=start_time,
#                                                data_variance = X_var,
#                                                data_mean=X_mean)
output = []
#for j in range(len(output2)):
#    output.append(output2[len(output2) - 1 - j][:].copy())
for j in range(x_data.shape[1]):
    output.append(x_data[0][j].copy())
for j in range(len(output1)):
    output.append(output1[j][:].copy())

nptensor_to_midi(output, send_file_mid)

#midi -> mp3
cmd = 'timidity -A 800 {0} -Ow -o - | lame - -b 64 {1}'. format(send_file_mid, send_file)
os.system(cmd)
#timidity received_music_mid.mid -Ow -o - | lame - -b 64 new.mp3
#/usr/local/share/timidity/timidity.cfg: No such file or directory
#timidity: Can't read any configuration file.
#Please check /usr/local/share/timidity/timidity.cfg


print('Finished generation!')
#        pdb.set_trace()
#gen wav file
#save_generated_example(send_file_wav, output, sample_frequency=44100)

#lame wav -> mp3
#cmd = 'lame -V2 {0} {1}'.format(send_file_wav, send_file)
#cmd = 'ffmpeg -i {0} -vn -ar 44100 -af "highpass=f=200, lowpass=f=3000" -f mp3 {1}'.format(send_file_wav, send_file)
#ffmpeg -i input.wav -vn -ar 44100 -ac 2 -ab 192k -f mp3 output.mp3

#os.system(cmd)