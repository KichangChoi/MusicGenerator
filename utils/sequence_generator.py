import numpy as np
import pdb
import random
import config.config_mg as nn_config

config = nn_config.get_neural_net_configuration()
num_dims = config['num_dims']

#Extrapolates from a given seed sequence
def generate_from_seed(model, seedSeq, sequence_length, thresh_hold, blank_ratio, average_note):
	output = []
	time_table = [2, 5, 10, 15, 20, 30, 40]
	now_note = [0 for row in range(80)]
	now_iter_note = []
	blank_percent = int(blank_ratio * 100)
	average_space = int(1 / average_note)


	seed_setting_length = 5

	for it in range(seed_setting_length * 20):
		dim3_seq_shape = (1, seedSeq.shape[1] - 1 - (seed_setting_length * 20 - it), num_dims)
		seedSeqSet = np.ones(dim3_seq_shape)
		seedSeqSet[0] = seedSeq[0][:seedSeq.shape[1] - 1 - (seed_setting_length * 20 - it)]
		seedSeqNew = model.predict(seedSeqSet)  # Step 1. Generate X_n + 1

		two_dim_result = seedSeqNew[0][seedSeqNew.shape[1] - 1].reshape(num_dims, 1)
		if it == 0:
			result = two_dim_result
		else:
			result = np.concatenate((result, two_dim_result), axis=1)

		if it % 20 == 0:
			print(str(it / 20) + 's seed setting networks')


	#The generation algorithm is simple:
	#Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
	#Step 2 - Concatenate X_n + 1 onto A
	#Step 3 - Repeat MAX_SEQ_LEN times
	for it in range(sequence_length * 20):
		seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
#		np.save('./fft_data_sets/new' + str(it), seedSeqNew)
		#Step 2. Append it to the sequence
#		if it == 0:
#			for i in range(seedSeqNew.shape[1]):#2nd dimension
#				output.append(seedSeqNew[0][i].copy())
#		else:
#		pdb.set_trace()

		# seedSeq = np.concatenate((seedSeq, seedSeqNew), axis=1)
		# np.mean(d, axis = 1)
		# np.std(d, axis = 1)
		#e.reshape(len(e), 1)
		#np.concatenate((b,e.reshape(4,1)), axis=1)
#		pdb.set_trace()

		one_dim_result = seedSeqNew[0][seedSeqNew.shape[1] - 1]
		two_dim_result = seedSeqNew[0][seedSeqNew.shape[1] - 1].reshape(num_dims, 1)

		result = np.concatenate((result, two_dim_result), axis=1)
		temp = (one_dim_result - np.mean(result, axis=1)) / np.std(result, axis=1)

		blank = temp[len(temp) - 1]
		temp[len(temp) - 1] = 0

		for i in range(len(now_note)):
			if now_note[i] != 0:
				for j in range(9):
					for k in range(7):
						temp[i + (80 * j) + (720 * k)] = 0
#			pdb.set_trace()
		if it < 5 or np.max(temp) <= 0:
			temp.fill(0)
			temp[len(temp) - 1] = 1
		else:
			max_index = np.argmax(temp)


			if blank_percent + (blank * 10) > random.randrange(1, 100):
				temp.fill(0)
				temp[len(temp) - 1] = 1
			else:
#					pdb.set_trace()
				multi_index = np.where(temp >= (temp[max_index] * thresh_hold))
				temp.fill(0)
				temp[max_index] = 1
				seleted_time = int(int(max_index) / 720)  # no, vel
				seleted_note = int(max_index) % 720 % 80
				now_iter_note.append(seleted_note)
				now_note[seleted_note] = time_table[seleted_time] + it
				for index in multi_index[0]:
					seleted_time = int(int(index) / 720)  # no, vel
					seleted_note = int(index) % 720 % 80
					for iter_note in now_iter_note:
						break_flag = 0
						if iter_note == seleted_note:
							break_flag = 1
							break

					if break_flag == 1:
						continue
					now_note[seleted_note] = time_table[seleted_time] + it
					now_iter_note.append(seleted_note)
					temp[index] = 1

			del now_iter_note[:]

		for i in range(len(now_note)):
			if now_note[i] <= it:
				now_note[i] = 0

#		pdb.set_trace()
		output.append(temp)

		newSeq = temp
		newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
		seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
		if it % 20 == 0:
			print(str(it / 20) + 's generated')

	#Finally, post-process the generated sequence so that we have valid frequencies
	#We're essentially just undo-ing the data centering process
#	for i in range(len(output)):
#		output[i] *= data_variance
#		output[i] += data_mean
	return output