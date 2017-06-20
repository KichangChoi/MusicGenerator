def get_neural_net_configuration():
	params = {}

	#80 notes, 9 velocity, 7 length of note, and no note
	params['num_dims'] = 80 * 9 * 7 + 1
	params['midi_file_tempo'] = 20
	params['sampling_frequency'] = 44100
	#Number of hidden dimensions.
	#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
	params['hidden_dimension_size'] = 1024
	#The weights filename for saving/loading trained models
	params['model_basename'] = './Weights_80iter'
	#The model filename for the training data
	params['model_file'] = './fft_data_sets/'
	#The dataset directory
	params['dataset_directory'] = '../youtube_downloader/music_rename/'
	return params