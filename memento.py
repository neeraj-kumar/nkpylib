"""Utilities to interact with the Memento database API."""
from __future__ import annotations

import os

import requests


def memento_api(endpoint, token='', **data):
    """Runs a memento API query to given `endpoint` with given `token`.

    If `token` is not given, then we use the `MEMENTO_ACCESS_TOKEN` env var.
    If you provide `data`, then we make a POST request, else GET.
    """
    if not token:
        token = os.environ['MEMENTO_ACCESS_TOKEN']
    base_url = 'https://api.mementodatabase.com/v1'
    assert endpoint.startswith('/')
    conjunction = '&' if '?' in endpoint else '?'
    url = base_url + endpoint + conjunction + 'token=%s' % (token)
    if data:
        r = requests.post(url, json=data)
    else:
        r = requests.get(url)
    return r.json()
