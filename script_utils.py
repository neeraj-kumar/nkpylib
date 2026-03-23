"""Some utils to help with writing scripts that run from the command line."""

from __future__ import annotations

import inspect
import logging

from argparse import ArgumentParser
from typing import Any, Callable

import yaml

from nkpylib.utils import specialize

logger = logging.getLogger(__name__)


def _cli_runner(func_list: list[Callable[..., Any]],
               description='',
               add_arbitrary=True,
               pre_func: Callable[..., Any]|None=None,
               **kw) -> Any:
    """Runs one of a given list of functions from the command line.

    If you give a `description` that is used for the argparser description. Else, we inspect the
    stack to get the caller's module name and docstring and create a generic description from those.

    By default, we allow the user to specify arbitrary key=value pairs on the command line to be
    passed to the function. You can disable this by setting `add_arbitrary` to False. The values
    are specialized using `nkpylib.utils.specialize`, so the user can pass in ints, floats, bools,
    lists, dicts, etc. (But note that occasionally this can bite you.)

    The `kw` arguments are used to define the command line arguments. Each key is the name of
    an argument, and the value is either a string (used as the help text) or a dict of keyword
    arguments to pass to `parser.add_argument`. If the first letter of the argument name
    hasn't been used yet, we add a short version of the argument using that letter. E.g. flag='set
    open flag' would create -f and --flag arguments, with help text 'set open flag'.

    For positional arguments, use the dict form of kw, and set the 'positional' key to True.

    If you provide a `pre_func`, that function is called with the parsed arguments (as **kwargs)
    prior to running the selected function. This can be used to set up logging, etc.

    This finally runs the selected function with the given arguments and returns the result.
    """
    funcs = {f.__name__: f for f in func_list}
    if not description:
        caller = inspect.stack()[1]
        module = inspect.getmodule(caller.frame)
        mod_name = module.__name__ if module else 'script'
        mod_doc = module.__doc__ if module and module.__doc__ else ''
        description = f'{mod_name} command line interface\n\n{mod_doc}'
    parser = ArgumentParser(description=description)
    parser.add_argument('func', choices=funcs, help=f"Function to run [{', '.join(funcs)}]")
    seen = ['h']
    for key, value in kw.items():
        cur_kw = {}
        if isinstance(value, str):
            cur_kw['help'] = value
        elif isinstance(value, dict):
            cur_kw.update(value)
        if cur_kw.pop('positional', False):
            parser.add_argument(key, **cur_kw)
        else:
            if key[0] in seen:
                parser.add_argument(f'--{key}', **cur_kw)
            else:
                parser.add_argument(f'-{key[0]}', f'--{key}', **cur_kw)
                seen.append(key[0])
    if add_arbitrary:
        parser.add_argument('keyvalue', nargs='*', help='Key=value pairs to pass to the function')
    args = parser.parse_args()
    kwargs = vars(args)
    for keyvalue in kwargs.pop('keyvalue', []):
        if '=' not in keyvalue:
            raise ValueError(f'Invalid key=value pair: {keyvalue}')
        key, value = keyvalue.split('=', 1)
        value = specialize(value)
        kwargs[key] = value
    if pre_func:
        pre_func(**kwargs)
    func = funcs[kwargs.pop('func')]
    return func(**kwargs) # type: ignore[operator]

def cli_runner(func_list: list[Callable[..., Any]], **kw) -> Any:
    """An alternative to my written one that uses argh"""
    #TODO see if we can allow arbitrary key=value commands...
    import argh
    parser = argh.ArghParser()
    parser.add_commands(func_list)
    parser.dispatch()


class NestedNamespace:
    """A namespace that supports nested attribute access."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, NestedNamespace(**value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Returns `None` for missing attributes instead of raising an attribute error"""
        return None

    def to_flat_dict(self, prefix: str = '', separator: str = '_') -> dict[str, Any]:
        """Convert nested namespace to a flat dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            full_key = f'{prefix}{key}' if prefix else key
            if isinstance(value, NestedNamespace):
                result.update(value.to_flat_dict(f'{full_key}{separator}', separator))
            else:
                result[full_key] = value
        return result


    def __repr__(self):
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                items.append(f'{key}=<NestedNamespace>')
            else:
                items.append(f'{key}={value!r}')
        return f"NestedNamespace({', '.join(items)})"


def load_yaml_configs(config_files: list[str] | None) -> dict:
    """Load and merge YAML config files.

    This takes yaml config files that can specify default values for any of the command-line
    arguments, and merges them together (with later files taking precedence) to produce a single
    config dictionary. This can then be used to set defaults for the command-line arguments.

    For nested configs, the top-level keys should correspond to the parser names (e.g., 'main' for
    the main parser), and the values should be dictionaries of argument defaults for that parser.
    """
    full_config = {}
    if config_files:
        for config_file in config_files:
            #logger.debug(f'Reading config file: {config_file}')
            with open(config_file) as f:
                file_config = yaml.safe_load(f)
                # Deep merge configs
                for key, value in file_config.items():
                    if key in full_config and isinstance(value, dict):
                        full_config[key].update(value)
                    else:
                        full_config[key] = value
    return full_config


class YamlConfigManager:
    """Context manager for YAML config handling with multiple parsers.

    Use this as follows:
    ```
        with YamlConfigManager() as config_manager:
            main_parser = config_manager.add_parser('main', description='Main parser')
            # Add arguments to main_parser
            sub_parser = config_manager.add_parser('sub', description='Sub parser')
            # Add arguments to sub_parser
            # When the context exits, the YAML config defaults will be applied to all parsers
        config = config_manager.parse_all()
        # now you can access config.main.arg1, config.sub.arg2, etc.
    """
    def __init__(self, cfg_path: str=''):
        self.config_parent = ArgumentParser(add_help=False)
        default = []
        if cfg_path:
            default.append(cfg_path)
        self.config_parent.add_argument('-c', '--configs', action='append', default=default, help='YAML config files')
        self.parsers: dict[str, ArgumentParser] = {}

    def add_parser(self, name: str, parser: ArgumentParser|None=None, **kwargs) -> ArgumentParser:
        """Add a parser with given `name` and `kwargs` to our config parent.

        If `parser` is given, we use that, else we create a new one with **kwargs

        """
        if 'parents' not in kwargs:
            kwargs['parents'] = []
        kwargs['parents'].append(self.config_parent)
        if parser is None:
            parser = ArgumentParser(**kwargs)
        self.parsers[name] = parser
        return parser

    def apply_yaml_defaults(self,
                            parsers: dict[str, ArgumentParser],
                            config_files: list[str] | None) -> None:
        """Apply YAML config defaults to multiple parsers.

        This sets the default values for those options from the yaml, meaning if you specify
        command line options for them, those will take precedence, but if you don't, the YAML values
        will be used. And if the yaml doesn't contain a value for an option, then the default
        specified in the argparser will be used.
        """
        full_config = load_yaml_configs(config_files)
        for section_name, parser in parsers.items():
            section_config = full_config.get(section_name, {})
            parser.set_defaults(**section_config)

    def parse_all(self, input_args=None) -> NestedNamespace:
        """Parse all parsers and return a nested namespace."""
        config_dict = {}
        for section_name, parser in self.parsers.items():
            logger.debug(f'parsing {section_name} with parser {parser}')
            args = parser.parse_args(args=input_args)
            # Remove shared arguments to avoid duplication
            section_dict = {k: v for k, v in vars(args).items() if k != 'configs'}
            config_dict[section_name] = section_dict
        return NestedNamespace(**config_dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parsers:
            temp_args, _ = self.config_parent.parse_known_args()
            logger.debug(f'Got parsers: {self.parsers}')
            self.apply_yaml_defaults(self.parsers, temp_args.configs)
