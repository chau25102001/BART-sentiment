from pathlib import Path
import json
from collections import OrderedDict
from utils.utils import read_json
import os


class ConfigParser:
    def __init__(self, args):
        self.config = read_json(args.config)
        save_dir = Path(self.config["save_dir"])
        self.config["save_dir"] = save_dir / self.config['name']
        os.makedirs(self.config['save_dir'], exist_ok=True)

    def init_obj(self, module, obj, *args, **kwargs):
        """
            initializing object with kwargs defined in config.json
            module: module object which contains the object,
            obj: key name of the object from config.json file. e.g.: "arch"
            *args: runtime argument that can not be defined in the config.json file
        """
        module_name = self.config[obj]['type']
        module_kwargs = self.config[obj]['kwargs']
        module_kwargs.update(kwargs)
        return getattr(module, module_name)(*args, **module_kwargs)

    def __getitem__(self, name):
        """Access items like an ordinary dict."""
        return self.config[name]
