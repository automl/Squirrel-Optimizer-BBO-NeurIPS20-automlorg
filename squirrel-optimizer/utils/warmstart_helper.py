import pickle
import numpy as np
import pandas as pd
from collections import Counter

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter

from inc_collection import load_config_spaces


def load_configs(df, model):
    top_configs = []
    if isinstance(df, pd.DataFrame):
        df.loc[model] = df.loc[model].dropna()
        for i, configs in enumerate(df.loc[model]):
            try:
                if type(configs) is not float:
                    top_configs.extend(configs)
            except Exception as e:
                print(e)
    else:  # if dict
        top_configs = df[model].to_numpy().tolist()
    return top_configs


def find_param_matches(df, param_name, param_info, config_spaces):
    matches = []
    if isinstance(df, pd.DataFrame):
        model_names = df.index
    else:  # if dict
        model_names = df.keys()
    for model in model_names:
        if model not in config_spaces.keys():
            print("Skipping since {} not found in loaded param spaces!".format(model))
            continue
        # looks for exact matches
        if param_name in config_spaces[model] and \
                param_info == config_spaces[model][param_name]:
            matches.append(model)
        # looks for partial matches differing only in their range (if applicable)
        elif param_name in config_spaces[model] and \
                param_info != config_spaces[model][param_name]:
            if 'type' in param_info and \
                    config_spaces[model][param_name]['type'] != param_info['type']:
                continue  # type of parameters should match
            if 'space' in param_info and \
                    config_spaces[model][param_name]['space'] != param_info['space']:
                continue  # the space of parameters should match (log/linear)
            if 'range' in param_info and \
                    config_spaces[model][param_name]['range'] != param_info['range']:
                matches.append(model)  # the ranges could be different if type, space are same
    return matches


def find_matches(df, api_config, path):
    matches = []
    config_spaces = load_config_spaces(path)
    for param_name, param_info in api_config.items():
        matches.extend(find_param_matches(df, param_name, param_info, config_spaces))

    # counts the number of parameters from a model matches api_config
    match_counter = Counter(matches)
    exact_match = []  # entire space matches exactly in all aspect ('range' can be different)
    partial_match = []  # differs by parameter name and meta information

    for key, v in match_counter.items():
        if v == len(api_config) and len(config_spaces[key]) == v:
            exact_match.append(key)
        elif v == len(api_config) and len(config_spaces[key]) < v:
            # partial_match.append(key)
            pass  # bypassing partial matches, ignoring this case
        # more than half the parameters match in name, type, space, (potentially) range
        elif v >= np.ceil(len(api_config) * 3 / 4).astype(int):
            # partial_match.append(key)
            pass  # bypassing partial matches, ignoring this case

    return exact_match, partial_match


def get_partial_configs(df, partial_match, api_config):
    inc_list = []
    for m in partial_match:
        config_list = load_configs(df, m)
        df_param = pd.DataFrame(config_list)
        df_api_config = pd.DataFrame(
            index=df_param.index, columns=list(api_config.keys())
        )
        for param_name, param_info in api_config.items():
            # loading values for matching parameters
            if param_name in df_param.columns:
                df_api_config[param_name] = df_param[param_name]
            # loading values for unmatched parameters
            else:
                param_type = param_info['type']
                if param_type == 'int':
                    low, high = param_info['range']
                    df_api_config[param_name] = np.random.randint(
                        low=low, high=high + 1, size=df_param.shape[0]
                    )
                elif param_type == 'real':
                    low, high = param_info['range']
                    df_api_config[param_name] = np.random.uniform(
                        low=low, high=high, size=df_param.shape[0]
                    )
                elif param_type == 'bool':
                    df_api_config[param_name] = np.random.choice(
                        [True, False], size=df_param.shape[0], replace=True
                    )
                elif param_type == 'cat':
                    df_api_config[param_name] = np.random.choice(
                        param_info['values'], size=df_param.shape[0], replace=True
                    )
        inc_list.extend([df_api_config.loc[i].to_dict() for i in df_api_config.index])
    return inc_list


def _clip_to_range(inc_list, api_config):
    df = pd.DataFrame(inc_list, columns=api_config.keys())
    for param_name, param_info in api_config.items():
        if param_info['type'] in ['real', 'int']:
            lower, upper = param_info['range']
            df[param_name] = df[param_name].clip(lower=lower, upper=upper)
        elif param_info['type'] in ['cat']:
            # replaces categories with choices available in current parameter space
            mismatches = [
                df[param_name].loc[i] not in param_info['values'] for i in df[param_name].index
            ]
            suggestions = np.random.choice(param_info['values'], size=sum(mismatches), replace=True)
            df[param_name][mismatches] = suggestions
    inc_list = [df.iloc[i].to_dict() for i in range(len(df))]
    return inc_list


def warmstart_load(df, api_config, path='./'):
    exact_match, partial_match = find_matches(df, api_config, path)
    if len(exact_match) > 0:
        inc_list = []
        for match in exact_match:
            configs = load_configs(df, match)
            for config in configs:
                exists = [config == inc for inc in inc_list]
                if not any(exists):
                    inc_list.extend([config])
        inc_list = _clip_to_range(inc_list, api_config)
        return inc_list
    elif len(partial_match) > 0:
        # sample non-nan values for the matching parameters
        # populate the rest randomly from its range
        inc_list = get_partial_configs(df, partial_match, api_config)
        inc_list = _clip_to_range(inc_list, api_config)
        return inc_list
    return None


def _sanitize_null_configs(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data).dropna()
        data = [data.loc[i].to_dict() for i in data.index]
        return data
    return data.dropna()


def get_cs_dimensions(api_config):
    """
	Help routine to setup ConfigurationSpace search space in constructor.
	Take api_config as argument so this can be static.
	Parameters
	----------
	api_config: Dict
		api dictionary to construct
	Returns
	-------
	cs: ConfigurationSpace
		ConfigurationSpace that contains the same hyperparameter as api_config
	"""
    # TODO 2 options to transform the real and int hyperaparameters in different scales
    #  option 1: similar to example_submission.skopt.optimizer, merge 'logit' into 'log' and 'bilog' into 'linear'
    #  option 2: use the api bayesmark.space.space to warp and unwarp the samples
    cs = ConfigurationSpace()
    param_list = sorted(api_config.keys())

    # hp_list = []
    for param_name in param_list:
        param_config = api_config[param_name]

        param_type = param_config["type"]
        param_space = param_config.get("space", None)
        param_values = param_config.get("values", None)
        param_range = param_config.get("range", None)

        if param_type == "cat":
            assert param_space is None
            assert param_range is None
            hp = CategoricalHyperparameter(name=param_name, choices=param_values)
        elif param_type == "bool":
            assert param_space is None
            assert param_values is None
            assert param_range is None
            hp = CategoricalHyperparameter(name=param_name, choices=[True, False])
        elif param_type == "ordinal":
            # appear in example_submission.skopt.optimizer but not in README
            assert param_space is None
            assert param_range is None
            hp = OrdinalHyperparameter(name=param_name, sequence=param_values)
        elif param_type in ("int", "real"):
            if param_values is not None:
                # TODO: decide whether we treat these parameters as discrete values
                #  or step function (example see example_submission.skopt.optimizer, line 71-77)
                # sort the values to store them in OrdinalHyperparameter
                param_values_sorted = np.sort(param_values)
                hp = OrdinalHyperparameter(name=param_name, sequence=param_values_sorted)
            else:
                log = True if param_space in ("log", "logit") else False
                if param_type == "int":
                    hp = UniformIntegerHyperparameter(name=param_name, lower=param_range[0],
                                                      upper=param_range[-1],
                                                      log=log)
                else:
                    hp = UniformFloatHyperparameter(name=param_name, lower=param_range[0],
                                                    upper=param_range[-1],
                                                    log=log)
        else:
            assert False, "type %s not handled in API" % param_type
        cs.add_hyperparameter(hp)

    return cs


def _dict_to_configspace(config_dict, cs, api_config):
    '''Converts types to match parameter space (cs)
    '''
    config = cs.sample_configuration()
    for param in config:
        if isinstance(cs.get_hyperparameter(param), UniformIntegerHyperparameter):
            config[param] = int(config_dict[param])
        elif isinstance(cs.get_hyperparameter(param), CategoricalHyperparameter):
            if api_config[param]['type'] == 'bool':
                # config_dict[param] will either be {0, 1} or {True, False}
                # int() type cast should work fine
                config[param] = \
                    cs.get_hyperparameter(param).choices[int(config_dict[param])]
            else:
                # simply reassign the string
                # assumption: a valid string from the 'choices' the parameter can take
                config[param] = config_dict[param]
        elif isinstance(cs.get_hyperparameter(param), OrdinalHyperparameter):
            config[param] = cs.get_hyperparameter(param).sequence[int(config_dict[param])]
        else:
            config[param] = config_dict[param]
    return config


def read_local_warmstarts(path, api_config, size=8, version=2, output_format='array'):
    '''Function to read local files and output a set of incumbents in chosen format
    Parameters
    ----------
    path : str
        The directory that houses the df.pkl file and the param_spaces/ folder
    api_config : dict
        The input parameter space as passed by Bayesmark
    size : int
        The number of configurations to suggest for first iteration
    output_format : str
        If output_format='list-dict', return each config as dicts inside a list
        If output_format='list-list', return each config as a list inside a list
        If output_format='list-configspace', return each config as a ConfigSpace
            object inside a list
        If output_format='array', return each config as a numpy array inside a numpy array
        If output_format='dataframe', return each config as a row of a dataframe
    Returns
    -------
    incumbents : list/array/dataframe
        Containing 'size' (or lesser) incumbents sampled from local files
    '''
    with open("{}/incumbents_v{}.pkl".format(path, version), "rb") as f:
        param_spaces = pickle.load(f)
    inc_list = warmstart_load(
        param_spaces, api_config, path="{}/param_spaces".format(path)
    )
    inc_list = _sanitize_null_configs(inc_list)

    if version == 3:
        select_idx = np.argsort([k['shapley'] for k in inc_list])[-min(size, len(inc_list)):]
        for i in inc_list:
            del i['shapley']
    elif version == 5:
        select_idx = np.arange(min(size, len(inc_list)))
    else:
        select_idx = np.random.choice(
            np.arange(len(inc_list)),
            size=min(size, len(inc_list)),  # can return lesser than 'size' configurations
            replace=False
        )

    incumbents = np.array(inc_list)[select_idx].tolist()

    if output_format == 'list-dict':
        pass
    elif output_format == 'list-list':
        incumbents = pd.DataFrame(incumbents).to_numpy().tolist()
    elif output_format == 'list-configspace':
        cs = get_cs_dimensions(api_config)
        incumbents = [_dict_to_configspace(config, cs, api_config) for config in incumbents]
    elif output_format == 'array':
        incumbents = pd.DataFrame(incumbents).to_numpy()
    elif output_format == 'dataframe':
        incumbents = pd.DataFrame(incumbents)
    else:
        legal_formats = {"list-dict", "list-list", "list-configspace", "array", "dataframe"}
        raise ValueError(
            "Invalid output_format selected, "
            "'{}' not among {}".format(output_format, legal_formats)
        )
    return incumbents
