"""Some utils to help with writing scripts run from the command line"""

from __future__ import annotations

import inspect

from argparse import ArgumentParser
from typing import Any, Callable

from nkpylib.utils import specialize

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
