import numpy as np
import tensorflow as tf
import pickle
import logging
from tqdm import tqdm
import os
from tensorflow.contrib import crf
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
	for i in range(len(dataLines)):
		line = dataLines[i]
		tokens = line.split()
		if i % 4 == 0:
			tag = [tagMap[token] for token in tokens] + [0]*(maxlen-len(tokens))
			sentences_tags.append(tag)
		elif i % 4 == 1:
			words = [word_dict[token] for token in tokens] + [0]*(maxlen-len(tokens))
			sentences.append(words)
		elif i % 4 == 2:
			poss = [posMap[token] for token in tokens] + [0]*(maxlen - len(tokens))
			sentences_pos.append(poss)
		else:
			ners = [nerMap[token] for token in tokens] + [0]*(maxlen - len(tokens))
			sentences_ner.append(ners)
			seq_len.append(len(tokens))
	return sentences, sentences_tags, seq_len, sentences_pos, sentences_ner


class Model(object):
	def __init__(self, hidden, num_layers, embedding, keep_prob, lr = 0.05, max_grad_norm = 10.0):
		self.hidden_size = hidden
		self.num_layers = num_layers
		self.embedding = embedding
		self.keep_prob = keep_prob
		self.lr = lr
		self.max_grad_norm = max_grad_norm
		self.build_model()

	def build_model(self):
		self.X_inputs = tf.placeholder(tf.int32, [None, maxlen], name='X_input')
		self.y_inputs = tf.placeholder(tf.int32, [None, maxlen], name='y_input')
		self.pos = tf.placeholder(tf.int32, [None, maxlen], name='pos')
		self.ner = tf.placeholder(tf.int32, [None, maxlen], name='ner')
		self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")

		with tf.device("/cpu:0"):
			inputs = tf.nn.embedding_lookup(self.embedding, self.X_inputs)
			inputs = tf.cast(inputs, dtype=tf.float32)
			pos = tf.one_hot(self.pos, len(posSet), axis=-1, dtype=tf.float32)
			ner = tf.one_hot(self.ner, len(nerSet), axis=-1, dtype=tf.float32)
			inputs = tf.concat([inputs, pos, ner], axis=2)

		m_cell_fw = []
		m_cell_bw = []
		for i in range(self.num_layers):
			cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
			cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
			cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
			cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
			m_cell_fw.append(cell_fw)
			m_cell_bw.append(cell_bw)

		batch = tf.shape(self.X_inputs)[0]
		m_cell_fw = tf.nn.rnn_cell.MultiRNNCell(m_cell_fw, state_is_tuple=True)
		m_cell_bw = tf.nn.rnn_cell.MultiRNNCell(m_cell_bw, state_is_tuple=True)
		init_state_fw = m_cell_fw.zero_state(batch, dtype=tf.float32)
		init_state_bw = m_cell_bw.zero_state(batch, dtype=tf.float32)
		output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=m_cell_fw, cell_bw=m_cell_bw, inputs=inputs,
														sequence_length=self.seq_len, scope='bi_lstm', dtype=tf.float32,
														initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
		output = tf.concat([output[0], output[1]], axis=2)
		logits = tf.layers.dense(output, len(tagMap))

		# logits = tf.nn.softmax(logits, dim=2)

		log_likelihood, transition_params = \
			crf.crf_log_likelihood(logits, self.y_inputs, self.seq_len)
		viterbi_sequence, viterbi_score = \
			crf.crf_decode(logits, transition_params, self.seq_len)
		self.cost = tf.reduce_mean(-log_likelihood)


		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_inputs, logits=logits)
		# [batch, time_step]
		seq_mask = tf.sequence_mask(self.seq_len, maxlen, tf.float32)
		# self.cost = tf.reduce_mean(seq_mask * loss)

		# self.y_ans = tf.cast(tf.argmax(logits, 2), tf.int32)
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

	def train(self, train_dset, test_dset, max_epoch):
		train_sentences, train_tags, train_seq_len, train_pos, train_ner = train_dset
		test_sentences, test_tags, test_seq_len, test_pos, test_ner = test_dset

		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		maxacc = 0
		saver = tf.train.Saver()
		for epoch in tqdm(range(max_epoch)):
			cost_val, train_op_val = sess.run([self.cost, self.train_op],
															feed_dict={self.X_inputs: train_sentences,
																	   self.y_inputs: train_tags,
																	   self.seq_len: train_seq_len,
																	   self.pos: train_pos,
																	   self.ner: train_ner})
			logging.info('epoch = %d, loss = %f' % (epoch, cost_val))
			ans, acc = sess.run([self.y_ans, self.accuracy], feed_dict={self.X_inputs: test_sentences, self.y_inputs: test_tags,
															  self.seq_len: test_seq_len,
															  self.pos: test_pos,
															  self.ner: test_ner})
			sent_acc_count = 0.0
			word_acc_count = 0.0
			word_count = 0.0
			for i in range(len(test_tags)):
				tags = test_tags[i]
				pred_true = True
				for j in range(test_seq_len[i]):
					word_count += 1.0
					if ans[i][j] != tags[j]:
						pred_true = False
					else:
						word_acc_count += 1.0
				if pred_true:
					sent_acc_count += 1.0
			logging.info('test acc: %f, word acc: %f, sentenceacc: %f' % (
			acc, word_acc_count / word_count, sent_acc_count / len(test_tags)))
			sent_acc = sent_acc_count / len(test_tags)
			if (sent_acc > maxacc):
				saver.save(sess, './model/model0.ckpt')


def main(hidden = 300, keep_prob = 0.95, num_layers = 1, lr = 0.05, max_grad_norm = 10.0, max_epoch = 100):
	with open('./data/word_dict.pickle', 'rb') as f:
		word_dict = pickle.load(f)
	with open('./data/embeddings.pickle', 'rb') as f:
		embeddings = pickle.load(f)
	train_dset = readfile_pos_ner('./data/train.phraselabel.nn.p', word_dict)
	test_dset = readfile_pos_ner('./data/test.phraselabel.nn.p', word_dict)
	model = Model(hidden, num_layers, embeddings, keep_prob, lr, max_grad_norm)
	model.train(train_dset, test_dset, 100)

if __name__ == '__main__':
	logging.basicConfig(filename='./log.txt', filemode='w', level=logging.DEBUG,
							format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
	main()