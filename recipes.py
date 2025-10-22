"""Various utilities related to recipes.

This deals with ld+json recipe cards, downloading recipes, parsing recipes, generating improved
recipes, etc.

TODO:
- remove $ amounts in ingredients
- split long steps into multiple steps
- embed substitutions in ingredients
  - look up from airtable
  - maybe url the substitute came from?
  - duplicate entire ingr structure, since quantities/notes/ might also change
- ingr table has "variants" text field, like "cherry tomato,roma tomato,beefsteak tomato"
  - embed each of these separately in recipe-texts, all linking back to the same ingr row
  - rather than ingredients having a substitution field (with self-refs), create a separate
    table with fields ingr1, notes1, ingr2, notes2, quantity, desc/misc notes
    - notes are things like "fresh", "small", or "cherry" to modify the key ingredient
- can use embeddings to get shortlist of similar, when looking up
- when asking llm, separate fields using ; and possibilities using |
- alternate quantities (e.g., 3 cups or 1 lb)?
  - still a full possibility, can merge in client somehow
  - maybe a "possibility type" field, like "quantity" or "unit" or "item", or multiple of those
- matching ingr -> step in client
  - split screen between steps and ingredients?
  - maybe just have an extra optional line below the step that shows ingr and quantities,
    clicking on which will jump you to the ingr
  - also have a voice command "show ingredients for current step"
- when we regenerate the recipe cards, re-use all-ingredients.json somehow?
- do analysis on ingredients
- pre-mixed/intermediate sets of ingredients like pastes or sauces
- figure out ways to use other recipe card fields, like video, comments, tags, yield, ratings,
  times, description, nutrition, categories, cuisines, dietary restrictions

- do some analysis on all recipe cards
    - all fields
    - field types with examples
- sectionizing:
  - For vegan banh xeo (ee146), the recipe card has no sections (all mixed together), and different from webpage
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from os.path import abspath
from subprocess import check_output
from typing import Any, Iterable, Optional, Union
from urllib.parse import urlparse

import unicodedata

from pyquery import PyQuery as pq # type: ignore

from nkpylib.constants import SIMPLE_URL_REGEXP
from nkpylib.ml.client import call_llm, get_text
from nkpylib.ml.llm_utils import llm_transform_list, batched_llm_call
from nkpylib.stringutils import GeneralJSONEncoder
from nkpylib.web_search import DefaultWebSearch
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

def fetch_recipe_from_url(url: str) -> Optional[dict[str, Any]]:
    """Tries to fetch a recipe from a url and returns the raw object.

    This can raise various errors.
    """
    logger.debug(f'Trying to pq {url}')
    s = make_request(url).text
    logger.debug(f'Fetched {len(s)} chars from {url}: {s[:200]}...')
    d = pq(s)
    logger.debug(f'Fetched {url}, title: {d("title").text()}')
    recipe = None
    # iterate through ld+json sections
    for s in d('script[type="application/ld+json"]'):
        logger.debug(f'  Found ld+json script tag: {s}')
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
    out = get_text.single(abspath(path))
    logger.debug(f'Extracted text from pdf ({len(out)} chars): {out[:200]}...')
    # extract urls from the text, normalizing them in various ways
    urls = [m.group(0) for m in SIMPLE_URL_REGEXP.finditer(out)]
    urls = ['https://' + url if not url.lower().startswith('http') else url for url in urls]
    urls = [unicodedata.normalize('NFKD', url).encode('ascii', 'ignore').decode('ascii') for url in urls]
    logger.debug(f'  Got {len(urls)} urls: {urls}')
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
    logger.debug(f'Getting url for recipe titled "{title}" from pdf {path}...{os.path.exists(path)}')
    if not os.path.exists(path):
        return ''
    checked_urls = set()
    checked_hosts = set()
    ws = DefaultWebSearch()
    for host, url in get_urls_from_pdf(path):
        logger.debug(f'  Host: {host}, URL: {url}')
        if url in checked_urls:
            break
        logger.debug(f'Checking url {url}')
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
        logger.debug(f'Searching for title "{title}" on host {host}')
        results = ws.search(title, site=host)
        for i, r in enumerate(results):
            logger.debug(f'    {i}: {json.dumps(asdict(r), indent=2)}\n')
            if not r.url:
                continue
            try:
                recipe = fetch_recipe_from_url(r.url)
                logger.debug(f'    Got recipe at {r.url}: {json.dumps(recipe, indent=2)[:500]}')
                if isinstance(recipe, dict):
                    return r.url
            except Exception as e:
                logger.warning(f'Error fetching recipe from {r.url}: {type(e)}: {e}')
                continue
    return ''


INGREDIENT_PROMPT = '''The following are a list of ingredients from a recipe for "NAME". Break each one into:
- a min and max quantity (e.g., 1, 2.5, 1/2), but converted to a float. If there's only a single
  value (e.g. 2), then min and max are the same (e.g. 2). If it's a range (e.g., 1-2), then min is
  the lower bound and max is the upper bound. Note that the unit might have a size, such as a 14 oz can, in which case the min and max quantities should refer to number of units (e.g. cans). If there's no value, then set these both to 1.
- a unit (e.g., cup, tbsp, tsp, g, kg, ml, l, oz, lb, etc.). If there's no unit (e.g., carrots or
  cucumber), then leave this blank.
- the full ingredient name as listed in the input (e.g., gluten-free flour, brown sugar, cherry
  tomatoes, etc.), but without any extra notes (see below)
- the ingredient main or primary name (e.g., flour, sugar, tomatoes, etc.)
- the importance of the ingredient: 2 if it's a key ingredient for the dish, 1 if it's a required
  but not key ingredient, and 0 if optional. For deciding if it's a key ingredient, use your best judgment based on the recipe name and other ingredients. In general, the main vegetables and distinctive spices are key (e.g. potatoes or sesame oil), standard things (like oil, salt or garlic) are not key, and garnishes (like scallions or cilantro) are sometimes optional.
- optionally, any additional notes (e.g., "finely chopped", "divided", "to taste", etc.)

For each input ingredient, output a JSON object with the following keys:
- min (float)
- max (float)
- unit (str)
- full_item (str)
- main_item (str)
- importance (0, 1, or 2)
- notes (list of str)

Strip any extra spaces, punctuation, parentheses, etc.

Some examples:

1/4 Cup Parmesan Cheese, grated, use more if prefer -> {"min":0.25,"max":0.25,"unit":"cup","full_item":"Parmesan Cheese","main_item":"Parmesan Cheese","importance":1,"notes":["grated", "use more if prefer"]}
4  garlic cloves (crushed) -> {"min":4,"max":4,"unit":"","full_item":"garlic cloves","main_item":"garlic","importance":1,"notes":["crushed"]}
2 14 ounces can fava beans -> {"min":2,"max":2,"unit":"14 ounces can","full_item":"fava beans","main_item":"fava beans","importance":2,"notes":[]}
1 (7 oz) block of Greek feta cheese in brine (torn into slabs) -> {"min":1,"max":1,"unit":"7 oz block","full_item":"Greek feta cheese in brine","main_item":"feta cheese","importance":1,"notes":["torn into slabs"]}
1 cup potatoes ((2 medium) chopped to \u00bd inch) -> {"min":1,"max":1,"unit":"cup","full_item":"potatoes","main_item":"potatoes","importance":2,"notes":["2 medium chopped to 1/2 inch"]}
\u00be to 1 cup cauliflower ((gobi florets)) -> {"min":0.75,"max":1,"unit":"cup","full_item":"cauliflower","main_item":"cauliflower","importance":2,"notes":["gobi florets"]}
1/2 cup thinly sliced green onion  ((optional, 2 stalks yield ~1/2 cup)) -> {"min":0.5,"max":0.5,"unit":"cup","full_item":"green onion","main_item":"green onion","importance":0,"notes":["2 stalks yield ~1/2 cup"]}
2 tbsp Olive Oil, replace 1 tbsp oil with sun dried tomato oil -> {"min":2,"max":2,"unit":"tbsp","full_item":"Olive Oil","main_item":"Olive Oil","importance":1,"notes":["replace 1 tbsp oil with sun dried tomato oil"]}
1/4 Cup Wine, good qaulity cooking wine, or leftover white wine -> {"min":0.25,"max":0.25,"unit":"cup","full_item":"Wine","main_item":"Wine","importance":1,"notes":["good qaulity cooking wine or leftover white wine"]}
garnishes/sides: handful of chopped fresh basil, or cilantro, optional sriracha or chili -> {"min":1,"max":1,"unit":"handful","full_item":"garnishes/sides","main_item":"garnish","importance":0,"notes":["handful of chopped fresh basil, or cilantro", "optional sriracha or chili"]}
'''

@dataclass
class Ingredient:
    """A class to represent an ingredient in a recipe."""
    id: str
    min: float|str
    max: float|str
    unit: str
    full_item: str
    main_item: str
    notes: list[str]
    importance: int = 1
    type: str = 'main'
    source: str = 'recipe'
    alternates: list[Ingredient]|None = None


class NKRecipe:
    """A class to represent an improved recipe card.
    """
    def __init__(self, recipe: dict[str, Any]) -> None:
        """Initialize this with an existing recipe card object."""
        self.recipe = recipe

    def __repr__(self) -> str:
        return f'<NKRecipe name="{self.recipe.get("name", "")}">'

    @classmethod
    def match_ingrs_to_steps(cls, all_steps: list[dict[str, Any]], ingrs: list[str], name: str) -> None:
        """For each step, output a list of matching ingredients, using an LLM"""
        ingrs = ['i_%d:%s' % (i, ingr.strip().replace('\n', '. ')) for i, ingr in enumerate(ingrs)]
        steps = []
        for i, step in enumerate(all_steps):
            try:
                steps.append(f's_%s:%s' % (step['section'], step['text'].replace("\n", ". ").strip()))
            except Exception as e:
                logger.warning(f'Error processing step {i}: {e} -> {step}')
                raise

        prompt = f'''Here is the full list of ingredients for the recipe "{name}", followed by the steps. Each ingredient is written as <ingredient id>:<ingredient text> and each step is written as <step index>. <step section>:<step text>

For each input step listed below, output a comma-separated list of ingredient ids that are used in that step. If there are no ingredients used in the step, output an empty string. If ingredients were used in a previous step and combined or cooked (e.g., to make a sauce or toasted, etc), don't include them in subsequent steps.

{len(ingrs)} Ingredients:
{ingrs}

Input Steps follow after this.
'''
        outputs = llm_transform_list(base_prompt=prompt, items=steps, chunk_size=50)
        # map values back to ingredient indices
        for cur, out in zip(all_steps, outputs):
            cur['ingredients'] = []
            if not out:
                continue
            cur['ingredients'] = [int(ingr_id.split('_',1)[-1]) for ingr_id in out.split(',')]

    @classmethod
    def add_timers_to_steps(cls, steps: list[dict[str, Any]]) -> None:
        """Adds timer information to steps."""
        inputs = [s['text'] for s in steps]
        prompt = f'''For each of the following steps of a recipe, determine if one or more durations are listed that would be helpful to run timers for. Output a (possibly-empty) comma-separated list of timers corresponding to each input step. There might be multiple timers per step. Output each timer in the form <short name>:<hours>:<minutes>:<seconds> where each of hours, minutes, and seconds might be 0. If a range is given, output a timer for both ends of the range, append " min" and " max" to the short name respectively.

Some examples:
Cover and let rest for 1 hour (or 30 minutes if you\u2019re in a hurry). -> rest min:0:30:0,rest max:1:0:0
Heat a medium or large skillet over medium-high heat (we recommend a well-seasoned cast iron). Once pan is hot, add the uncooked tortilla and cook until the edges start to lift away (about 30-45 seconds). Then flip and cook an additional 30-45 seconds. -> cook min:0:0:30,cook max:0:0:45,flip min:0:0:30,flip max:0:0:45
Leftovers can be stored covered in the refrigerator for 2-3 days or the freezer for 1 month. Best when fresh. Reheat in the microwave. -> 
If using chili flakes, add them at this moment. Sizzle for 10 seconds or so. Then add light soy sauce, Shaoxing rice wine, Chinese five spice powder, and sugar. Cook for a further 30 seconds or so. -> sizzle:0:0:10,cook:0:0:30
Heat a large pot over medium-low heat. Once hot, add oil (or water) and garlic. Saut\u00e9 briefly for 1 minute, stirring frequently, until barely golden brown. Then add tomatoes, oregano, coconut sugar, salt, and pepper flake. -> saute:0:1:0
If you\u2019re using canned chickpeas, drain and rinse. If using our Instant Pot Chickpeas, simply ensure your chickpeas are drained of excess cooking liquid and proceed as instructed. -> 
To a large mixing bowl add chickpeas, harissa paste, lemon juice, minced garlic, salt, maple syrup, paprika, and olive oil and stir gently to combine. -> 
Bring a pot of water to boil and add a tablespoon of natural coarse salt. Add chopped potatoes and carrots and boil until just cooked (about 12 minutes). Strain, and set aside until needed.\u00a0 -> boil:0:12:0

Input Steps follow after this.
'''
        outputs = llm_transform_list(base_prompt=prompt, items=inputs, chunk_size=50)
        for cur, out in zip(steps, outputs):
            cur['timers'] = []
            if not out:
                continue
            for timer in out.split(','):
                timer = timer.strip()
                if timer:
                    name, h, m, s = timer.split(':')
                    cur['timers'].append(dict(name=name, h=int(h), m=int(m), s=int(s)))

    @classmethod
    def transform_ingredients(cls, ingredients: list[str], name: str) -> list[Ingredient]:
        """Transforms our list of ingredients using an LLM."""
        ret = []
        # run llm in parallel across all ingredients
        llm_outs = batched_llm_call(prompt=INGREDIENT_PROMPT.replace('NAME', name),
                                    inputs=ingredients,
                                    batch_size=50,
                                    json_outputs=True,
                                    model='llama4')
        counts_by_name: Counter[str] = Counter()
        for batch, outputs in llm_outs:
            for input, obj in zip(batch, outputs):
                if not obj:
                    raise ValueError(f'Error processing ingredient: {input}')
                if isinstance(obj, str):
                    raise ValueError(f'Ingredient output is string, not object: {input} -> {obj}')
                name = obj['main_item'].lower().replace(' ', '_')[:3]
                counts_by_name[name] += 1
                obj.setdefault('notes', [])
                obj.setdefault('importance', 1)
                ingr = Ingredient(id=f'{name}{counts_by_name[name]}', **obj)
                #print(f' input: "{input}" -> {obj} -> {ingr}')
                ret.append(ingr)
        return ret

    @classmethod
    def clean_text(cls, s: str|Any) -> str|Any:
        """Does some cleanup of text, such as replacing &nbsp; with spaces, stripping spaces, etc.

        Note that if the input is not a string, we return it as-is
        """
        if not isinstance(s, str):
            return s
        s = html.unescape(s)
        try:
            s = re.sub(r'\s+', ' ', s).strip()
        except Exception as e:
            logger.warning(f'Error cleaning text {e}: {type(s)}: {s}')
        return s

    @classmethod
    def sectionize_steps(cls, instructions: list[dict[str,Any]]) -> dict[str, list[dict[str, Any]]]:
        """Splits the ingredients from the raw recipe into sections as needed. Also does some cleanup."""
        # if we have no sections, enclose in a section named main
        ret = {}
        if isinstance(instructions, list) and len(instructions) > 0 and isinstance(instructions[0], str):
            instructions = [{'itemListElement': instructions, 'name': 'main', '@type': 'HowToSection'}]
        if isinstance(instructions, str):
            instructions = [{'itemListElement': [instructions], 'name': 'main', '@type': 'HowToSection'}]
        try:
            if instructions[0]['@type'] == 'HowToStep':
                instructions = [{'itemListElement': instructions, '@type': 'HowToSection', 'name': 'main'}]
        except Exception as e:
            logger.warning(f'Error checking for sections: {e} -> {instructions}')
            raise
        done = set()
        for section in instructions:
            name = section['name']
            secname = name.lower()[:4]
            num = 1
            while f's-{secname}{num}' in done:
                num += 1
            basename = f's-{secname}{num}'
            done.add(basename)
            steps = [s['text'] if 'text' in s else s for s in section['itemListElement']]
            ret[name] = [dict(text=cls.clean_text(s), section=name, id=f'{basename}-{i+1}00') for i, s in enumerate(steps)]
        return ret

    def improve(self) -> dict[str, Any]:
        """Improves our recipe and outputs a new recipe card object."""
        recipe = deepcopy(self.recipe)
        # transform ingredients
        name = recipe.get('name', recipe.get('title', recipe.get('headline', '')))
        orig_ingrs = [self.clean_text(ingr) for ingr in recipe.pop("recipeIngredient")]
        recipe['ingredients'] = {'main': self.transform_ingredients(orig_ingrs, name=name)}
        ingr_id_by_idx = {i: ingr.id for i, ingr in enumerate(recipe['ingredients']['main'])}
        #TODO figure out how to split ingredients into sections
        # split steps into sections and do some cleanup
        recipe['steps'] = self.sectionize_steps(recipe.pop("recipeInstructions"))
        all_steps = []
        for sec, steps in recipe['steps'].items():
            all_steps.extend(steps)
        self.match_ingrs_to_steps(all_steps=all_steps,
                                  ingrs=orig_ingrs,
                                  name=name)
        self.add_timers_to_steps(all_steps)
        # map the recipe step-ingredients to ingredient ids
        for sec, steps in recipe['steps'].items():
            for step in steps:
                step['ingredients'] = [ingr_id_by_idx[i] for i in step['ingredients'] if i in ingr_id_by_idx]
        #print(json.dumps(recipe, indent=2, cls=GeneralJSONEncoder))
        return recipe
