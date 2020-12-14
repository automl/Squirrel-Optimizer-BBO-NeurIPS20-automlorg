######################################################################################
# Reads locally found incumbents to build a DataFrame for quick querying and mapping #
######################################################################################

import os
import sys
import pickle
import numpy as np
import pandas as pd


def load_config_spaces(path='misc/'):
    files = os.listdir(path)
    config_spaces = {}
    for filename in files:
        if 'api' in filename:
            with open('{}/{}'.format(path, filename), 'rb') as f:
                cs = pickle.load(f)
            model = filename.split('_')[0]
            config_spaces[model] = cs
    return config_spaces


if __name__ == '__main__':
	version = sys.argv[1]
	path = '/'.join(__file__.split('/')[:-2])
	base_dir = os.path.join(path, 'utils', 'final_positions')
	print(path)
	models = []
	datasets = []
	for f in os.listdir(base_dir):
		# if "v{}".format(version) not in f:
		# 	continue
		df = pd.read_csv(f"{base_dir}/{f}", sep=';', index_col = 0, decimal = ',')
		models.append(f.split('.')[0].split('_')[-1])
		if len(datasets) == 0:
			for d in np.unique(df.dataset):
				datasets.append(d)

	param_spaces = load_config_spaces(path=os.path.join(path, 'utils', 'param_spaces'))
	main_df = pd.DataFrame([], index=models, columns=datasets)

	for f in os.listdir(base_dir):
		df = pd.read_csv(f"{base_dir}/{f}", sep=';', index_col = 0, decimal = ',')
		model = f.split('.')[0].split('_')[-1]
		param_list = list(param_spaces[model].keys())
		for d in np.unique(df.dataset):
			df_subset = df.loc[df.dataset == d]
			df_subset = [df_subset.loc[i, list(param_list)].to_dict() for i in df_subset.index]
			main_df.loc[model, d] = df_subset
		print(main_df.loc[model])

	with open(os.path.join(path, 'utils', 'incumbents_v{}.pkl'.format(version)), 'wb') as f:
		pickle.dump(main_df, f)

	print("Exiting inc_collection...")