import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
import config.config_mg as nn_config
from midi.MidiOutFile import MidiOutFile

def nptensor_to_midi(x_data, out_filename):
	midi = MidiOutFile(out_filename)

	config = nn_config.get_neural_net_configuration()
	model_basename = config['model_basename']
	hidden_dims = config['hidden_dimension_size']
	num_dims = config['num_dims']

	# non optional midi framework
	midi.header(format=0, nTracks=1, division=10)
	midi.start_of_track()

	midi.tempo(500000)

	# musical events

	no = 0
	vel = 0
	last_event_time = 0
	now_playing_note = []
	num_seq_len = len(x_data)
	time_table = [2, 5, 10, 15, 20, 30, 40]

	for i in range(num_seq_len):
		note_vector_index = np.argwhere(x_data[i]==1)

		if x_data[i][num_dims - 1] == 1:
			no = 0
		else:
			for one_note in note_vector_index:
				time = int(one_note / 720)#no, vel
				vel = (int(one_note % 720 / 80) + 1) * 10
				no = int(one_note) % 720 % 80 + 21
				midi.update_time(last_event_time)
				midi.note_on(channel=0, note=no, velocity=vel)
				last_event_time = 0
				now_playing_note.append([i, no, time_table[time]])

		for now_note in now_playing_note:
			if i - now_note[0] == now_note[2]:
				midi.update_time(last_event_time)
				midi.note_off(channel=0, note=now_note[1])
				last_event_time = 0
				del now_note


		last_event_time += 1



	# non optional midi framework
	midi.update_time(0)

	midi.end_of_track()  # not optional!

	midi.eof()

	print('good')



def convert_mp3_to_wav(filename, sample_frequency):
	ext = filename[-4:]
	if(ext != '.mp3'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-4]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in range(len(files)-1):
		new_path += files[i]+'/'
	tmp_path = new_path + 'tmp'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
	new_name = new_path + '/' + orig_filename + '.wav'
	sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
	cmd = 'lame -a -m m {0} {1}'.format(filename, filename_tmp)
	os.system(cmd)
	cmd = 'lame --decode {0} {1} --resample {2}'.format(filename_tmp, new_name, sample_freq_str)
	os.system(cmd)
	return new_name

def convert_flac_to_wav(filename, sample_frequency):
	ext = filename[-5:]
	if(ext != '.flac'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-5]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in range(len(files)-1):
		new_path += files[i]+'/'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	new_name = new_path + '/' + orig_filename + '.wav'
	cmd = 'sox {0} {1} channels 1 rate {2}'.format(quote(filename), quote(new_name), sample_frequency)
	os.system(cmd)
	return new_name


def convert_folder_to_wav(directory, sample_rate=44100):
	for file in os.listdir(directory):
		fullfilename = directory+file
		if file.endswith('.mp3'):
			convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)
		if file.endswith('.flac'):
			convert_flac_to_wav(filename=fullfilename, sample_frequency=sample_rate)
	return directory + 'wave/'

def read_wav_as_np(filename):
	data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
	#np_arr = np.array(np_arr)
	return np_arr, data[0]

def write_np_as_wav(X, sample_rate, filename):
	Xnew = X * 32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	num_samples_so_far = 0
	while(num_samples_so_far < total_samples):
		block = song_np[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0],))
			block = np.concatenate((block, padding))
		block_lists.append(block)
		num_samples_so_far += block_size
	return block_lists

def convert_sample_blocks_to_np_audio(blocks):
	song_np = np.concatenate(blocks)
	return song_np

def time_blocks_to_fft_blocks(blocks_time_domain):
	fft_blocks = []
	for block in blocks_time_domain:
		fft_block = np.fft.fft(block)
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks	

def fft_blocks_to_time_blocks(blocks_ft_domain):
	time_blocks = []
	for block in blocks_ft_domain:
		num_elems = block.shape[0] / 2
		real_chunk = block[0:num_elems]
		imag_chunk = block[num_elems:]
		new_block = real_chunk + 1.0j * imag_chunk
		time_block = np.fft.ifft(new_block)
		time_blocks.append(time_block)
	return time_blocks

def convert_wav_files_to_nptensor(file, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
	many_files = []
	X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
	chunks_X = []
	chunks_Y = []
	cur_seq = 0
	total_seq = len(X)
	print (file)
#	print (max_seq_len)
	while cur_seq + max_seq_len < total_seq:
		chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
		chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
		cur_seq += max_seq_len

	num_examples = len(chunks_X)
	num_dims_out = block_size * 2

	out_shape = (num_examples, max_seq_len, num_dims_out)
	x_data = np.zeros(out_shape)
	y_data = np.zeros(out_shape)
	for n in range(num_examples):
		for i in range(max_seq_len):
			x_data[n][i] = chunks_X[n][i]
			y_data[n][i] = chunks_Y[n][i]
		if n == len(num_examples):
			print ('Saved example ', (n+1), ' / ',num_examples)
	print ('Flushing to disk...')
	#편차 데이터 읽어오기


	mean_x = np.load('./fft_data_sets/mean.npy')
	std_x = np.load('./fft_data_sets/std.npy')

	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	np.save(out_file + '/x', x_data)
	np.save(out_file + '/y', y_data)
	print ('Done!')

def convert_wav_files_to_nptensor_service(file, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
	many_files = []
	X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
	chunks_X = []
	chunks_Y = []
	total_seq = len(X)
	print (file)

	num_dims_out = block_size * 2

	out_shape = (1, total_seq, num_dims_out)
	x_data = np.zeros(out_shape)
	y_data = np.zeros(out_shape)
	for i in range(total_seq):
		x_data[0][i] = X[i]
		y_data[0][i] = Y[i]

	print ('Flushing to disk...')
	#편차 데이터 읽어오기


	mean_x = np.load('./fft_data_sets/mean.npy')
	std_x = np.load('./fft_data_sets/std.npy')

	x_data[0][:] -= mean_x #Mean 0
	x_data[0][:] /= std_x #Variance 1
	y_data[0][:] -= mean_x #Mean 0
	y_data[0][:] /= std_x #Variance 1

	np.save(out_file + '/x', x_data)
	np.save(out_file + '/y', y_data)
	print ('Done!')


'''
	for file in os.listdir(directory):
		if file.endswith('.wav'):
			many_files.append(directory+file)
	chunks_X = []
	chunks_Y = []
#	num_files = len(many_files)
	num_files = 1
	if(num_files > max_files):
		num_files = max_files
	for files in many_files:
		for file_idx in range(num_files):
			file = files
			print ('Processing: ', (file_idx+1),'/',num_files)
			print ('Filename: ', file)
'''

def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
	num_seqs = tensor.shape[1]
	for i in indices:
		chunks = []
		for x in range(num_seqs):
			chunks.append(tensor[i][x])
		save_generated_example(filename+str(i)+'.wav', chunks,useTimeDomain=useTimeDomain)

def load_training_example(filename, block_size=2048, useTimeDomain=False):
	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size)) #Add special end block composed of all zeros
	if useTimeDomain:
		return x_t, y_t
	X = time_blocks_to_fft_blocks(x_t)
	Y = time_blocks_to_fft_blocks(y_t)
	return X, Y

def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
	if useTimeDomain:
		time_blocks = generated_sequence
	else:
		time_blocks = fft_blocks_to_time_blocks(generated_sequence)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, sample_frequency, filename)
	return

def audio_unit_test(filename, filename2):
	data, bitrate = read_wav_as_np(filename)
	time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
	ft_blocks = time_blocks_to_fft_blocks(time_blocks)
	time_blocks = fft_blocks_to_time_blocks(ft_blocks)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, bitrate, filename2)
	return
