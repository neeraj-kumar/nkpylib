"""Module to deal with external providers of ML services."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil

from argparse import ArgumentParser

from nkpylib.ml.constants import PROVIDERS_PATH
from nkpylib.web_utils import make_request_async
from nkpylib.thread_utils import sync_or_async

logger = logging.getLogger(__name__)

def iter_providers():
    """Iterates through (provider_name, provider) pairs."""
    with open(PROVIDERS_PATH) as f:
        providers = json.load(f)
    for name, provider in providers.items():
        yield name, provider


async def call_provider(provider_name, endpoint, headers_kw=None, files=None, **data):
    """Call a provider's openai-compatible API at given `endpoint`.

    If `data` is provided, then we make a POST request, else GET.
    By default, we set a content type of application/json in the header and set the BEARER token
    appropriate for the provider. If you provide your own `headers_kw`, we add to those, unless they
    have one of these keys, in which case we use the provided value.

    You can optionally provide `files` to upload files to the provider. In that case, we don't
    explicitly set a content-type.
    """
    provider = {name: provider for name, provider in iter_providers()}[provider_name]
    logger.debug(f'Calling {provider_name} at {endpoint}')
    token = os.environ.get(provider['api_key_var'])
    headers = {
        "Authorization": f"Bearer {token}",
    }
    if not files:
        headers["Content-Type"] = "application/json"
    if endpoint.startswith('http'): # it's the full url, so don't add the base_url
        url = endpoint
    else:
        url = provider['base_url'] + endpoint
    if headers_kw:
        headers.update(headers_kw)
    headers = {k: v for k, v in headers.items() if v}
    req_kw = dict(url=url, headers=headers, min_delay=0)
    if data:
        if files:
            ret = await make_request_async(method='post', files=files, **req_kw)
        elif 'json' in headers.get('Content-Type', ''):
            ret = await make_request_async(method='post', json=data, **req_kw)
        else:
            ret = await make_request_async(method='post', data=data, **req_kw)
    else:
        ret = await make_request_async(method='get', **req_kw)
    return ret.json()

async def call_external(endpoint, headers_kw=None, provider_name='', **data):
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
    ret = await call_provider(provider_name, endpoint, headers_kw, **data)
    return ret


@sync_or_async
async def update_providers(path=PROVIDERS_PATH):
    """Update the providers file with the latest data for each provider."""
    with open(path) as f:
        providers = json.load(f)
    for name, provider in providers.items():
        logger.info(f'\nUpdating provider {name}')
        if provider.get('no_model_api'):
            continue
        _models = await call_provider(name, '/models')
        models = _models['data']
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
