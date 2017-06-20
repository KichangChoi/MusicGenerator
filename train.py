from __future__ import absolute_import
from __future__ import print_function
from midi.MidiInFile import MidiInFile
from midi.MidiReadList import MidiReadList # the event handler
from midi.MidiOutFile import MidiOutFile
import numpy as np
from datetime import datetime
import utils.network_utils as network_utils
from parse_files import *
import config.config_mg as nn_config
import utils.sequence_generator as sequence_generator


config = nn_config.get_neural_net_configuration()
input_directory = config['dataset_directory']
output_filename = config['model_file']

freq = config['sampling_frequency']  # sample frequency in Hz
clip_len = 10  # length of clips for training. Defined in seconds
block_size = freq / 4  # block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size))  # Used later for zero-padding song sequences

inputFile = config['model_file']
cur_iter = 0
model_basename = config['model_basename']
model_filename = config['model_file']

# Load up the training data
print('Loading training data')
# X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# X_train = np.load(inputFile + '_x.npy')
# y_train = np.load(inputFile + '_y.npy')
# print ('Finished loading training data')

# Figure out how many frequencies we have in the data
# freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']
num_dims = config['num_dims']
# Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=num_dims,
                                          num_hidden_dimensions=hidden_dims)
# model_reverse = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
# You could also substitute this with a RNN or GRU
# model = network_utils.create_gru_network()

# Load existing weights if available
if os.path.isfile(model_basename):
#	model.load_weights(model_filename)
    model.load_weights(model_basename)
# model_reverse.load_weights(model_basename + '_reverse')

num_iters = 100  # Number of iterations for training
epochs_per_iter = 10  # Number of iterations before we save our model
batch_size = 16  # Number of training examples pushed to the GPU per batch.
# Larger batch sizes require more memory, but training will be faster
print('Starting training!')
no_files = 603
cur_iter = 0
while cur_iter < num_iters:
    f = open("result.txt", 'a')
    f.write('====================================================')
    f.write('Forward iter : ' + str(cur_iter) + '  |  ')
    f.write(str(datetime.today().strftime("%H-%M-%S")))
    f.write('\n\n')
    f.close()

    for no in range(1, 603):
        if no == 221 or no == 425 or no == 445:
            continue

        if no / 10 < 1:
            filename = '00' + str(no)
        else:
            if no / 100 < 1:
                filename = '0' + str(no)
            else:
                filename = str(no)
        # get data
        test_file = 'data_sets/wave/mid/'+ filename + '.mid'

        # do parsing

        midiIn = MidiInFile(MidiReadList(), test_file)
        midiIn.read()

        midi_list = []
        midi_list = midiIn.parser.dispatch.outstream.note_list

        all_seq_len = midiIn.parser.dispatch.outstream.alltime
        max_seq_len = 200
        num_examples = int(all_seq_len / max_seq_len)

        vector_0 = np.zeros(num_dims)

        out_shape = (num_examples, max_seq_len, num_dims)

        x_data = np.zeros(out_shape)
        y_data = np.zeros(out_shape)

        for i in range(len(midi_list)):
            if max_seq_len * num_examples <= midi_list[i][0]:
                continue
            x_data[int(midi_list[i][0] / 200)][int(midi_list[i][0] % 200)][midi_list[i][1] + (midi_list[i][2] * 80) + (midi_list[i][3] * 720)] = 1
            if int(midi_list[i][0] % 200) != 0:
                y_data[int(midi_list[i][0] / 200)][int(midi_list[i][0] % 200) - 1][midi_list[i][1] + (midi_list[i][2] * 80) + (midi_list[i][3] * 720)] = 1

        for i in range(num_examples):
            for j in  range(max_seq_len):
                if np.array_equal(vector_0, x_data[i][j]):
                    x_data[i][j][num_dims - 1] = 1
                    if j != 0:
                        y_data[i][j-1][num_dims - 1] = 1

    #    y_data[0] = x_data[0][1:]
    #    np.append(y_data[0], np.zeros((1, num_dims)), axis=0)  # Add special end block composed of all zeros
        # while cur_iter < num_iters:
        #	print('Iteration: ' + str(cur_iter))
        # We set cross-validation to 0,
        # as cross-validation will be on different datasets
        # if we reload our model between runs
        # The moral way to handle this is to manually split
        # your data into two sets and run cross-validation after
        # you've trained the model for some number of epochs
        history = model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1,
                            validation_split=0.0)

        f = open("result.txt", 'a')
        f.write(str(cur_iter) + "/" + str(num_iters) + " file " + str(no) + " Training complete!  |  ")
        f.write(str(datetime.today().strftime("%H-%M-%S")))
        f.write("\n\n")
        f.close()

        del midi_list[:]
        del midiIn
        del x_data
        del y_data
        del vector_0

    cur_iter += epochs_per_iter
    model.save_weights(model_basename, overwrite=True)


