import pickle
import pandas as pd
import torch

def pickle_to_file(file_name, data, protocol = pickle.HIGHEST_PROTOCOL):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol)


def unpickle_from_file(file_name):
	with open(file_name, 'rb') as handle:
		try:
			return pickle.load(handle)
		except ImportError:
			return pd.read_pickle(file_name)

def save_training_history(file_name_training_history, logger, training_history):

	if logger is not None:
		logger.log('Saving the training history to: ' + file_name_training_history)

	pickle_to_file(file_name_training_history, training_history)


def save_network(file_name_model, logger, model):

	if logger is not None:
		logger.log('Saving the model to: ' + file_name_model)

	torch.save(model.state_dict(), file_name_model)


def load_network(file_name_model, logger, model):

	if logger is not None:
		logger.log('Loading the model from: ' + file_name_model)

	model.load_state_dict(torch.load(file_name_model))