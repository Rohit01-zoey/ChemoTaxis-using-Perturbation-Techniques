import yaml

global cfg
if 'cfg' not in globals():
    with open('./src/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


def get_cfg():
    """Return the global config object."""
    return cfg

