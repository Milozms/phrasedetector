import numpy as np
import tensorflow as tf
import pickle
import logging
from tqdm import tqdm
import os
from tensorflow.contrib import crf
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

maxlen = 18
tagMap = {
	'N': 0,
	'V-B': 1,
	'V-I': 2,
	'R-B': 3,
	'R-I': 4,
	'E-B': 5,
	'E-I': 6,
	'C-B': 7,
	'C-I': 8,
	'A-B': 9,
	'A-I': 10
}
posSet = ['padding', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
		  'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
		  'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
		  'WDT', 'WP', 'WP$', 'WRB']
nerSet = ['padding', 'GPE-I', 'FACILITY-I', 'ORGANIZATION-I', 'FACILITY-B', 'GPE-B',
		  'NE', 'PERSON-I', 'ORGANIZATION-B', 'GSP-B', 'PERSON-B']

def build_map(s):
	posMap = {}
	for i, tag in enumerate(s):
		posMap[tag] = i
	return posMap

def lstm_cell(num_units, keep_prob=0.5):
	cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
	return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

def readfile(filename, word_dict):
	f = open(filename, 'r')
	sentences = []
	sentences_tags = []
	seq_len = []
	dataLines = f.readlines()
	for i in range(len(dataLines)):
		line = dataLines[i]
		tokens = line.split()
		if i % 2 == 0:
			tag = [tagMap[token] for token in tokens] + [0]*(maxlen-len(tokens))
			sentences_tags.append(tag)
		else:
			words = [word_dict[token] for token in tokens] + [0]*(maxlen-len(tokens))
			sentences.append(words)
			seq_len.append(len(tokens))
	return sentences, sentences_tags, seq_len

def readfile_pos_ner(filename, word_dict):
	posMap = build_map(posSet)
	nerMap = build_map(nerSet)
	f = open(filename, 'r')
	sentences = []
	sentences_tags = []
	sentences_pos = []
	sentences_ner = []
	seq_len = []
	dataLines = f.readlines()
	dset = []
	instance = {}
	for i in range(len(dataLines)):
		line = dataLines[i]
		tokens = line.split()
		if i % 4 == 0:
			tag = [tagMap[token] for token in tokens] + [0]*(maxlen-len(tokens))
			# sentences_tags.append(tag)
			instance['tag'] = tag
		elif i % 4 == 1:
			words = [word_dict[token] for token in tokens] + [0]*(maxlen-len(tokens))
			# sentences.append(words)
			instance['words'] = words
		elif i % 4 == 2:
			poss = [posMap[token] for token in tokens] + [0]*(maxlen - len(tokens))
			# sentences_pos.append(poss)
			instance['pos'] = poss
		else:
			ners = [nerMap[token] for token in tokens] + [0]*(maxlen - len(tokens))
			# sentences_ner.append(ners)
			instance['ner'] = ners
			# seq_len.append(len(tokens))
			instance['len'] = len(tokens)
			dset.append(instance)
			instance = {}
	return dset

class Model(object):
	def __init__(self, hidden, num_layers, embedding, keep_prob, lr = 0.05, max_grad_norm = 10.0, batch_size = 64):
		self.hidden_size = hidden
		self.num_layers = num_layers
		self.embedding = embedding
		self.keep_prob = keep_prob
		self.lr = lr
		self.posdim = 100
		self.nerdim = 100
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size
		self.build_model()

	def build_model(self):
		self.X_inputs = tf.placeholder(tf.int32, [None, maxlen], name='X_input')
		self.y_inputs = tf.placeholder(tf.int32, [None, maxlen], name='y_input')
		self.pos = tf.placeholder(tf.int32, [None, maxlen], name='pos')
		self.ner = tf.placeholder(tf.int32, [None, maxlen], name='ner')
		self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")

		with tf.device("/cpu:0"):
			word_emb = tf.nn.embedding_lookup(self.embedding, self.X_inputs)
			word_emb = tf.cast(word_emb, dtype=tf.float32)
			pos_embedding = tf.get_variable("pos_embedding",
											shape=[len(posSet), self.posdim],
											trainable=True,
											dtype=tf.float32,
											initializer=tf.random_normal_initializer())
			ner_embedding = tf.get_variable("ner_embedding",
											shape=[len(nerSet), self.nerdim],
											trainable=True,
											dtype=tf.float32,
											initializer=tf.random_normal_initializer())
			# pos = tf.one_hot(self.pos, len(posSet), axis=-1, dtype=tf.float32)
			# ner = tf.one_hot(self.ner, len(nerSet), axis=-1, dtype=tf.float32)
			pos = tf.nn.embedding_lookup(pos_embedding, self.pos)
			ner = tf.nn.embedding_lookup(ner_embedding, self.ner)
			inputs = tf.concat([word_emb, pos, ner], axis=2)


		batch = tf.shape(self.X_inputs)[0]
		hidden = self.hidden_size
		keep_prob = self.keep_prob
		seq_len = self.seq_len

		with tf.variable_scope('biLSTM'):
			outputs = [inputs]
			for layer in range(self.num_layers):
				with tf.variable_scope("Layer_{}".format(layer)):
					inputs_fw = outputs[-1]
					with tf.variable_scope("fw"):
						cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden)
						cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, input_keep_prob=keep_prob,
																output_keep_prob=keep_prob)
						init_fw = cell_fw.zero_state(batch, dtype=tf.float32)
						out_fw, state_fw = tf.nn.dynamic_rnn(
							cell_fw, inputs_fw, sequence_length=seq_len, initial_state=init_fw, dtype=tf.float32)
					with tf.variable_scope("bw"):
						inputs_bw = tf.reverse_sequence(
							inputs_fw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
						cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden)
						cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, input_keep_prob=keep_prob,
																output_keep_prob=keep_prob)
						init_bw = cell_bw.zero_state(batch, dtype=tf.float32)
						out_bw, state_bw = tf.nn.dynamic_rnn(
							cell_bw, inputs_bw, sequence_length=seq_len, initial_state=init_bw, dtype=tf.float32)
						out_bw = tf.reverse_sequence(
							out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
					outputs.append(tf.concat([out_fw, out_bw], axis=2))
			output = tf.concat(outputs[1:], axis=2)

		# output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=m_cell_fw, cell_bw=m_cell_bw, inputs=inputs,
		# 												sequence_length=self.seq_len, scope='bi_lstm', dtype=tf.float32,
		# 												initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
		# output = tf.concat([output[0], output[1]], axis=2)
		output = tf.layers.dense(output, hidden, activation=tf.nn.tanh)
		output = tf.layers.dense(output, len(tagMap))
		self.unary_scores = output

		with tf.variable_scope("crf"):
			log_likelihood, transition_params = \
				crf.crf_log_likelihood(self.unary_scores, self.y_inputs, self.seq_len)
			viterbi_sequence, viterbi_score = \
				crf.crf_decode(self.unary_scores, transition_params, self.seq_len)
			tf.get_variable_scope().reuse_variables()
			self.transition_params = tf.get_variable("transitions")

		self.cost = tf.reduce_mean(-log_likelihood)


		# [batch, time_step]
		seq_mask = tf.sequence_mask(self.seq_len, maxlen, tf.float32)

		self.y_ans = viterbi_sequence
		correct_prediction = tf.equal(self.y_ans, self.y_inputs)  # [batch, time_step]
		correct_cnt = tf.reduce_sum(seq_mask * tf.cast(correct_prediction, tf.float32), axis=1)
		# [batch]
		accuracy = correct_cnt / tf.cast(self.seq_len, tf.float32)
		self.accuracy = tf.reduce_mean(accuracy)

		tvars = tf.trainable_variables()
		# 获取损失函数对于每个参数的梯度
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
		# 优化器
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		# 梯度下降计算
		self.train_op = optimizer.apply_gradients(zip(grads, tvars),
											 global_step=tf.contrib.framework.get_or_create_global_step())
		return

	def test(self, test_dset):
		test_sentences = [instance['words'] for instance in test_dset]
		test_seq_len = [instance['len'] for instance in test_dset]
		test_pos = [instance['pos'] for instance in test_dset]
		test_ner = [instance['ner'] for instance in test_dset]

		tf_unary_scores, transition_params = sess.run([self.unary_scores, self.transition_params],
							feed_dict={self.X_inputs: test_sentences,
									   self.seq_len: test_seq_len,
									   self.pos: test_pos,
									   self.ner: test_ner})
		sent_acc_count = 0.0
		word_acc_count = 0.0
		word_count = 0.0
		test_tags = [instance['tag'] for instance in test_dset]

		for i in range(len(test_tags)):
			unary_score = tf_unary_scores[i][:test_seq_len[i]]
			viterbi_sequence, viterbi_score = \
				crf.viterbi_decode(unary_score, transition_params)
			tags = test_tags[i]
			pred_true = True
			for j in range(test_seq_len[i]):
				word_count += 1.0
				if viterbi_sequence[j] != tags[j]:
					pred_true = False
				else:
					word_acc_count += 1.0
			if pred_true:
				sent_acc_count += 1.0
		logging.info('word acc: %f, sentenceacc: %f' % (
			word_acc_count / word_count, sent_acc_count / len(test_tags)))
		sent_acc = sent_acc_count / len(test_tags)
		return sent_acc

	def train(self, train_dset, test_dset, max_epoch):
		batch_size = self.batch_size
		train_datasize = len(train_dset)
		num_batch = int(math.ceil(train_datasize/batch_size))



		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		maxacc = 0
		saver = tf.train.Saver()
		for epoch in tqdm(range(max_epoch)):
			cost = 0.0
			for nbatch in range(num_batch):
				if nbatch == num_batch - 1:
					batch_dset = train_dset[nbatch*batch_size:]
				else:
					batch_dset = train_dset[nbatch*batch_size: (nbatch+1)*batch_size]
				train_sentences = [instance['words'] for instance in batch_dset]
				train_tags = [instance['tag'] for instance in batch_dset]
				train_seq_len = [instance['len'] for instance in batch_dset]
				train_pos = [instance['pos'] for instance in batch_dset]
				train_ner = [instance['ner'] for instance in batch_dset]
				cost_val, train_op_val = sess.run([self.cost, self.train_op],
															feed_dict={self.X_inputs: train_sentences,
																	   self.y_inputs: train_tags,
																	   self.seq_len: train_seq_len,
																	   self.pos: train_pos,
																	   self.ner: train_ner})
				cost += cost_val*len(batch_dset)
			cost /= len(train_dset)
			logging.info('Epoch = %d, loss = %f' % (epoch, cost))
			sent_acc = self.test(test_dset)
			if (sent_acc > maxacc):
				# saver.save(sess, './model/model0.ckpt')
				maxacc = sent_acc
			np.random.shuffle(train_dset)
		logging.info('max acc: %f' % maxacc)
		print('max acc: %f' % maxacc)
		return maxacc


def main(hidden = 200, keep_prob = 0.8, num_layers = 1, lr = 0.025, max_grad_norm = 10.0, max_epoch = 100, batch_size = 128):
	with open('./data/word_dict.pickle', 'rb') as f:
		word_dict = pickle.load(f)
	with open('./data/embeddings.pickle', 'rb') as f:
		embeddings = pickle.load(f)
	train_dset = readfile_pos_ner('./data/train.phraselabel.nn.p', word_dict)
	test_dset = readfile_pos_ner('./data/test.phraselabel.nn.p', word_dict)
	model = Model(hidden, num_layers, embeddings, keep_prob, lr, max_grad_norm, batch_size)
	model.train(train_dset, test_dset, max_epoch)

def crossvalid(hidden = 200, keep_prob = 0.8, num_layers = 1, lr = 0.025, max_grad_norm = 10.0, max_epoch = 100, batch_size = 64):
	with open('./data/word_dict.pickle', 'rb') as f:
		word_dict = pickle.load(f)
	with open('./data/embeddings.pickle', 'rb') as f:
		embeddings = pickle.load(f)
	dset = readfile_pos_ner('./data/train.phraselabel.nn.p', word_dict) \
		   + readfile_pos_ner('./data/test.phraselabel.nn.p', word_dict)
	np.random.shuffle(dset)
	numfold = math.floor(len(dset) / 5)
	acc = 0.0
	for i in range(0, 5):
		logging.info('Fold %d:' % i)
		test_dset = dset[numfold * i: numfold * (i + 1)]
		train_dset = dset[:numfold * i] + dset[numfold * (i + 1):]
		with tf.variable_scope("Fold_{}".format(i)):
			model = Model(hidden, num_layers, embeddings, keep_prob, lr, max_grad_norm, batch_size)
			fold_acc = model.train(train_dset, test_dset, max_epoch)
			acc += fold_acc
		logging.info('......')
		logging.info('\n\n')
	acc /= 5
	print('Average accuracy: %f' % acc)


if __name__ == '__main__':
	logging.basicConfig(filename='./log.txt', filemode='w', level=logging.DEBUG,
							format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
	main()
	# crossvalid()