from itertools import product


def get_all_combinations(config_space):
    keys, values = zip(*config_space.items())

    all_combinations = []
    for combination in product(*values):
        config = dict(zip(keys, combination))
        all_combinations.append(config)

    return all_combinations
