"""Module to deal with external providers of ML services."""

from __future__ import annotations

import json
import logging
import os
import shutil

from argparse import ArgumentParser

import requests

from constants import PROVIDERS_PATH

logger = logging.getLogger(__name__)

def iter_providers():
    """Iterates through (provider_name, provider) pairs."""
    with open(PROVIDERS_PATH) as f:
        providers = json.load(f)
    for name, provider in providers.items():
        yield name, provider


def call_provider(provider_name, endpoint, headers_kw=None, **data):
    """Call a provider's API at given `endpoint`.

    If `data` is provided, then we make a POST request, else GET.
    By default, we set a content type of application/json in the header and set the BEARER token
    appropriate for the provider. If you provide your own `headers_kw`, we add to those, unless they
    have one of these keys, in which case we use the provided value.
    """
    provider = {name: provider for name, provider in iter_providers()}[provider_name]
    logger.debug(f'Calling {provider_name} at {endpoint}')
    token = os.environ.get(provider['api_key_var'])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    url = provider['base_url'] + endpoint
    if headers_kw:
        headers.update(headers_kw)
    if data:
        ret = requests.post(url, headers=headers, json=data).json()
    else:
        ret = requests.get(url, headers=headers).json()
    return ret

def call_external(endpoint, headers_kw=None, provider_name='', **data):
    """Call an external API at given `endpoint`.

    If you give a `provider_name`, we use that provider (with no checking of model name validity).
    If you don't, we try to find the right provider based on the 'model' key in `data`.

    If `data` is provided, then we make a POST request, else GET.
    By default, we set a content type of application/json in the header and set the BEARER token
    appropriate for the provider. If you provide your own `headers_kw`, we add to those, unless they
    have one of these keys, in which case we use the provided value.
    """
    logger.debug(f'Calling external API at {endpoint} with provider {provider_name} and data {data}')
    if not provider_name:
        if 'model' not in data:
            raise ValueError('You must provide a provider_name, or a model in the data')
        model = data['model']
        for name, provider in iter_providers():
            logger.debug(f'checking provider {name} for model {model}, {model in provider["models"]}')
            if model in provider['models']:
                provider_name = name
                break
        else:
            raise ValueError(f'No provider found for model {model}')
    return call_provider(provider_name, endpoint, headers_kw, **data)


def update_providers(path=PROVIDERS_PATH):
    """Update the providers file with the latest data for each provider."""
    with open(path) as f:
        providers = json.load(f)
    for name, provider in providers.items():
        logger.info(f'\nUpdating provider {name}')
        if provider.get('no_model_api'):
            continue
        models = call_provider(name, '/models')['data']
        logger.debug(f'  {len(models)} models: {models[:10]}')
        provider['models'] = {m['id']: m for m in models}
    # backup existing one to %s.bak (overwriting any previous), then write new one
    shutil.copy(path, path + '.bak')
    with open(path, 'w') as f:
        json.dump(providers, f, indent=2) # don't sort keys, as we want to iterate in order!


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)
    parser = ArgumentParser()
    funcs = {f.__name__: f for f in [update_providers]}
    parser.add_argument('func', choices=funcs.keys(), help='function to run')
    args = parser.parse_args()
    func = funcs[args.func]
    func()
