"""Various utilities related to recipes.

This deals with ld+json recipe cards, downloading recipes, parsing recipes, generating improved
recipes, etc.


TODO:
- Recipe card++
  - ingredients:
    - generate id -> deterministic, like "gar1"?
    - range: always have min, max and if it's one value, they're just the same (front-end displays
      differently)
    - embed substitutions in ingredients
      - look up from airtable
      - rather than ingredients having a substitution field (with self-refs), create a separate
        table with fields ingr1, notes1, ingr2, notes2, quantity, desc/misc notes
        - notes are things like "fresh", "small", or "cherry" to modify the key ingredient
    - ingr has bool field "is key ingredient" (for that recipe)
    - ingr table has "variants" text field, like "cherry tomato,roma tomato,beefsteak tomato"
      - embed each of these separately in recipe-texts, all linking back to the same ingr row
    - can use embeddings to get shortlist of similar, when looking up
    - ingr unit has size (small, medium, 14 oz can, etc)
    - also "actual item name": "tomato"
    - complicated example: "1 (7 oz) block of Greek feta cheese in brine (torn into slabs)"
    - item has possibilities, like coconut oil or olive oil
      - these are like substitutes, and should be represented the same, so maybe each alternate has
        a field like "alternate source", which is "from recipe" or "from airtable" or "from bing"
      - maybe a url the substitute came from?
      - duplicate entire ingr structure, since quantities/notes/ might also change
    - when asking llm, separate fields using ; and possibilities using |
    - default unit "" (e.g., carrots)
    - alternate quantities (e.g., 3 cups or 1 lb)?
      - still a full possibility, can merge in client somehow
      - maybe a "possibility type" field, like "quantity" or "unit" or "item", or multiple of those
    - "garnishes/sides: handful of chopped fresh basil, or cilantro, optional sriracha or chili
      garlic sauce"
  - matching ingr -> step
    - maybe don't have sections be a dict because it might lose order?
    - prompt: "here are all [original string, not parsed] ingredients with ids. for each input step,
      output comma-separated ingredient ids, if any"
    - inputs: each step, with all newlines removed
    - in client: split screen between steps and ingredients?
      - maybe just have an extra optional line below the step that shows ingr and quantities,
        clicking on which will jump you to the ingr
      - also have a voice command "show ingredients for current step"
    - don't include pastes, intermediate things
  - when we regenerate the recipe cards, we can re-use all-ingredients.json somehow
  - do analysis on ingredients
  - output notes separately for ingredients
  - Consider ranges and substitutes for ingredients.
  - Steps:
    - Include metadata and linkages to quantities and pre-mixed sets of ingredients.
  - make sure to include recipe yield to modify

- what about intermediate pastes/sauces?

- figure out ways to use other recipe card fields, like video, comments, tags, yield, ratings,
  times, description, nutrition, categories, cuisines, dietary restrictions

- do some analysis on all recipe cards
    - all fields
    - field types with examples
- parse recipe cards into cleaned up/reformatted recipe cards
  - For vegan banh xeo (ee146), the recipe card has no sections (all mixed together), and different
    from webpage

"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from subprocess import check_output
from typing import Any, Iterable, Optional, Union
from urllib.parse import urlparse

from pyquery import PyQuery as pq # type: ignore

from nkpylib.constants import URL_REGEXP
from nkpylib.ml.client import call_llm
from nkpylib.ml.llm_utils import llm_transform_list
from nkpylib.web_search import BingWebSearch
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

def fetch_recipe_from_url(url: str) -> Optional[dict[str, Any]]:
    """Tries to fetch a recipe from a url and returns the raw object.

    This can raise various errors.
    """
    d = pq(url=url, opener=lambda url, **kw: make_request(url, **kw).text)
    recipe = None
    # iterate through ld+json sections
    for s in d('script[type="application/ld+json"]'):
        # try to JSON load it directly
        try:
            x = json.loads(s.text)
        except Exception:
            continue
        # check for the recipe object in various places
        to_check = []
        # first in @graph (most common)
        if "@graph" in x:
            to_check.extend(x["@graph"])
        # then check the object itself
        if isinstance(x, list): # it might be a list of objects
            for obj in x:
                if isinstance(obj, dict):
                    to_check.append(obj)
        elif isinstance(x, dict): # or it might be an object itself
            to_check.append(x)
        for obj in to_check:
            types = obj.get("@type", "")
            if isinstance(types, str):
                types = [types]
            if "recipe" not in {t.lower() for t in types}:
                continue
            recipe = obj
        if recipe is not None:
            break
    return recipe

def get_urls_from_pdf(path: str) -> Iterable[tuple[str, str]]:
    """Yields (host, url) pairs extracted from a pdf.

    These are sorted by most common hostname.
    """
    urls_by_host = defaultdict(set)
    args = ["pdftotext", path, "-"]
    out = check_output(args).decode("utf-8", "replace")
    urls = [m.group(0) for m in URL_REGEXP.finditer(out)]
    # group links by hostname
    for url in urls:
        urls_by_host[urlparse(url).hostname].add(url)
    hosts = Counter(urlparse(url).hostname for url in urls)
    for host, _count in hosts.most_common():
        if host is None:
            continue
        for url in urls_by_host[host]:
            yield host, url

def get_url_from_recipe(path: str, title: str) -> str:
    """Gets the url from a recipe, or empty string on error.

    This iterates through all urls from a recipe pdf, trying to fetch the recipe card from that
    url. It also tries to search for the recipe based on the host (domain) of the url and the given
    `title`. Again, if it can successfully parse a recipe card, then it returns that url.

    It returns the first url that matches, or '' if none do.
    """
    if not os.path.exists(path):
        return ''
    checked_urls = set()
    checked_hosts = set()
    ws = BingWebSearch()
    for host, url in get_urls_from_pdf(path):
        if url in checked_urls:
            break
        checked_urls.add(url)
        # try parsing the recipe directly from this url
        try:
            r = fetch_recipe_from_url(url)
            if isinstance(r, dict):
                return url
        except Exception as e:
            logger.warning(f'Error fetching {url}: {e}')
            continue
        # now try searching for the url based on the title
        if host in checked_hosts:
            continue
        checked_hosts.add(host)
        results = ws.search(f'site:{host} {title}')
        for i, r in enumerate(results):
            logger.debug(f'    {i}: {json.dumps(r, indent=2)}\n')
            if not r['url']:
                continue
            try:
                recipe = fetch_recipe_from_url(r['url'])
                logger.debug(f'    Got recipe at {r["url"]}: {json.dumps(recipe, indent=2)[:500]}')
                if isinstance(recipe, dict):
                    return r['url']
            except Exception as e:
                logger.warning(f'Error fetching recipe from {r["url"]}: {type(e)}: {e}')
                continue
    return ''


INGREDIENT_PROMPT = '''The following are a list of ingredients from a recipe. Break each one into:
- a quantity (e.g., 1, 2.5, 1/2), but converted to a float
- a unit (e.g., cup, tbsp, tsp, g, kg, ml, l, oz, lb, etc.)
- an ingredient name (e.g., flour, sugar, salt, etc.)
- optionally, any additional notes (e.g., "finely chopped", "divided", "to taste", etc.)

For each input ingredient, output the elements above separated by semicolons, all in one line.
'''

Ingredient = dict[str, Union[float, str, list[str]]]

class NKRecipe:
    """A class to represent an improved recipe card.
    """
    def __init__(self, recipe: dict[str, Any]) -> None:
        """Initialize this with an existing recipe card object."""
        self.recipe = recipe

    def __repr__(self) -> str:
        return f'<NKRecipe name="{self.recipe.get("name", "")}">'

    def transform_ingredients(self, ingredients: list[str]) -> list[Ingredient]:
        """Transforms our list of ingredients using an LLM.

        """
        out_ingr = []
        # run llm in parallel
        outputs = llm_transform_list(base_prompt=INGREDIENT_PROMPT, items=ingredients, chunk_size=20)
        for input, output in zip(ingredients, outputs):
            if not output:
                raise ValueError(f'Error processing ingredient: {input}')
            out = [el.strip() for el in output.split(';')]
            qty: Union[float, str]
            qty, unit, item, *notes = out
            notes = [n.strip() for n in notes if n.strip()]
            try:
                qty = float(qty)
            except ValueError:
                pass
            print(f' input: "{input}" -> output: "{qty};{unit};{item};{" ".join(notes)}"')
            out_ingr.append(dict(quantity=qty, unit=unit, item=item, notes=notes))
        return out_ingr

    def improve(self) -> dict[str, Any]:
        """Improves our recipe and outputs a new recipe card object."""
        recipe = deepcopy(self.recipe)
        # transform ingredients
        recipe['ingredients'] = {'main': self.transform_ingredients(recipe.pop("recipeIngredient"))}
        #TODO figure out how to split ingredients into sections
        # steps might come with sections or not
        recipe['steps'] = {}
        instructions = recipe.pop('recipeInstructions')
        # if we have no sections, enclose in a section named main
        if instructions[0]['@type'] == 'HowToStep':
            instructions = [{'itemListElement': instructions, '@type': 'HowToSection', 'name': 'main'}]
        for section in instructions:
            name = section['name']
            steps = [s['text'] if 'text' in s else s for s in section['itemListElement']]
            recipe['steps'][name] = steps
        return recipe
