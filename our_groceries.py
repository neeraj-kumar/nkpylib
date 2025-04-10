"""Interface to deal with OurGroceries.

This is reverse engineered from the website. I actually couldn't figure out how it gets the initial
page content, as no matter what I tried, I just couldn't see the initial list coming down anywhere
(html, js, xhr, etc).

However, fortunately, if you add an item or remove an item, the response is the full list.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time

from typing import Any

from scipy.spatial.distance import cdist

from nkpylib.web_utils import make_request
from nkpylib.ml.client import embed_text

logger = logging.getLogger(__name__)

# Base URL for all API calls
BASE_URL = 'https://www.ourgroceries.com/your-lists'

# Common keyword arguments for all API calls
COMMON_KW = dict(listId='VmVrdSIkgMUdxXglxBOlfk',
                 teamId='INtqvCJrPC6hhJq7jSQIea',
                 shareId=None,
                 locale='en-US')

# typedef for the list of items
Item = dict[str, Any]
GroceryList = list[Item]

class OurGroceries:
    """Class to interact with OurGroceries API."""

    def __init__(self):
        """Reads our list and maintains it"""
        self.embeddings = {}
        self.items = self.get_list(include_deleted=True)

    def _api(self, command: str,
                   headers: dict | None = None,
                   method='POST',
                   include_deleted=False,
                   **kw) -> dict:
        """Call the OurGroceries API with the given `command`, returning (response, cur list of items).

        We will automatically add various headers if not given, including the auth token (gotten from
        the cookie 'ourgroceries-auth').

        This will parse out the current list from the response JSON data and return it as the 2nd element,
        with the 1st element being the response JSON data without the list.

        If you set `include_deleted` to True, then we will include crossed off items in the list.

        This is a lower-level function, you probably want to use the higher-level functions like
        `add_item` or `remove_item`.
        """
        auth = os.environ.get('OUR_GROCERIES_AUTH')
        common_headers = {
            'Accept': 'application/json',
            'Accept-Language': 'en-US',
            'Content-Type': 'application/json; charset=UTF-8',
            'Cookie': f'ourgroceries-auth={auth}; g_addMultipleItems=true',
        }
        actual_headers = dict(common_headers)
        if headers:
            actual_headers.update(headers)
        data = dict(command=command, **COMMON_KW, **kw)
        r = make_request(url=BASE_URL, method=method, headers=actual_headers, json=data)
        #print(r.request.__dict__)
        obj = r.json()
        assert 'list' in obj, f'No list in response: {obj}'
        self.items = self.parse_list(obj.pop('list')['items'], include_deleted=include_deleted)
        self.update_embeddings()
        return obj

    def update_embeddings(self):
        """Updates our embeddings for the items in the list."""
        todo = [el['name'] for el in self.items if el['name'] not in self.embeddings]
        embeddings = embed_text.batch(todo)
        self.embeddings.update(dict(zip(todo, embeddings)))

    def parse_list(self, lst: list[dict], include_deleted: bool = False) -> GroceryList:
        """Parses the response JSON data from any OurGroceries API call and returns a list of items.

        The outputs are dicts that are from the API, except that we extract out the quantity from the name
        and put it in a separate key 'quantity', replacing the name with the name without the quantity.
        (The 'value' field is the original name with quantity.)
        """
        # go in order, excluding crossed off items
        if not include_deleted:
            lst = [i for i in lst if not i.get('crossedOffAt')]
        for i in lst:
            if i['name'] != i['value']:
                logger.warning(f"List name '{i['name']}' != value '{i['value']}'")
            # also parse out quantities which are appended to the item name as (qty)
            m = re.match(r'(.+)\s+\((\d+)\)', i['value'])
            if m:
                i['name'] = m.group(1)
                i['quantity'] = int(m.group(2))
            else:
                i['quantity'] = 1
        return lst

    # RAW API CALLS
    def add_item(self, item: str, **kw) -> dict:
        """Add the given `item` to the grocery list."""
        return self._api(command='insertItem', value=item, isFromRecipe=False, **kw)

    def check_item(self, item_id: str, **kw) -> dict:
        """Marks the given `item_id` from the grocery list as done (checked-off)."""
        return self._api(command='setItemCrossedOff', crossedOff=True, itemId=item_id, **kw)

    def uncheck_item(self, item_id: str, **kw) -> dict:
        """Marks the given `item_id` from the grocery list as not done (unchecked)."""
        return self._api(command='setItemCrossedOff', crossedOff=False, itemId=item_id, **kw)

    def delete_item(self, item_id: str, **kw) -> dict:
        """Deletes the given `item_id` from the grocery list completely."""
        return self._api(command='deleteItem', itemId=item_id, **kw)

    def change_item(self, item_id: str, new_item: str, **kw) -> dict:
        """Changes the given item to the new item."""
        return self._api(command='changeItemValue', newValue=new_item, itemId=item_id, **kw)


    # PROPERTIES (computed)
    @property
    def active(self) -> GroceryList:
        """List of active items in the grocery list (not checked off)."""
        return [i for i in self.items if not i.get('crossedOffAt')]


    # HIGHER-LEVEL API CALLS
    def get_list(self, include_deleted=False) -> GroceryList:
        """Get the current grocery list.

        Note that I haven't figured out a clean way to do this, so we add a dummy item then remove it.
        """
        obj = self.add_item(f'nk-dummy')
        #print(f'Added dummy item: {obj}')
        self.delete_item(obj['itemId'], include_deleted=include_deleted)
        return self.items

    def item_id_by_name(self, item: str) -> str | None:
        """Returns the item ID of the item with the given name (possibly excluding quantity), or None if not found."""
        lst = self.get_list(include_deleted=True)
        print(f'Got lst with {len(lst)} items: {lst[:5]}...{lst[-5:]}')
        for i in lst:
            if i['name'] == item or i['value'] == item:
                return i['id']
        return None

    def __contains__(self, item: str) -> bool:
        """Returns True if the given `item` (name) is in the list and not checked off."""
        return self.item_id_by_name(item) is not None

    def get_closest(self, name: str, k: int=10) -> list[tuple[float, str]]:
        """Returns the closest items to the given `name`.

        This first checks if the `name` is an exact match to any of the items in the list, and if
        so returns that. Else, computes distances using our embeddings and returns the top `k`.

        Returns pairs of (distance, name).
        """
        if name in self:
            return [(0.0, name)]
        name_emb = embed_text.single(name)
        items, embs = zip(*self.embeddings.items())
        dists = cdist([name_emb], embs, 'cosine')[0]
        # sort by distance
        idxs = dists.argsort()[:k]
        ret = [(dists[i], items[i]) for i in idxs]
        return ret

    def add_closest(self, name: str, threshold: float=0.1) -> Item:
        """Adds the closest item to the given `name` to the list if it is not already in the list.

        If it's already there and checked off, we uncheck it.
        If it's already there and not checked off, we do nothing.
        """
        dist, match = self.get_closest(name, k=1)[0]
        logger.info(f'Adding closest item to {name}: {dist} {match}')
        if dist <= threshold:
            item_id = self.item_id_by_name(match)
            if item_id:
                [item] = [el for el in self.items if el['id'] == item_id]
                if item['crossedOffAt']:
                    self.uncheck_item(item_id)
                return item
        else: # just add it
            itemId = self.add_item(name)
        [item] = [item for item in self.items if item['id'] == itemId]
        return item

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d: %(message)s')
    og = OurGroceries()
    print(og.items[:5], og.items[-5:])  # print first and last 5 items
    for arg in sys.argv[1:]:
        close = og.get_closest(arg)
        print(f'Finding similar to "{arg}": {json.dumps(close, indent=2)}')
    og.add_item('Flonase')
    og.add_item('tofu')
