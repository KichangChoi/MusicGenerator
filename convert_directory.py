from parse_files import *
import config.nn_config as nn_config

config = nn_config.get_neural_net_configuration()
input_directory = config['dataset_directory']
output_filename = config['model_file'] 

freq = config['sampling_frequency'] #sample frequency in Hz
clip_len = 10 		#length of clips for training. Defined in seconds
block_size = freq / 10 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
#Step 1 - convert MP3s to WAVs
new_directory = convert_folder_to_wav(input_directory, freq)
