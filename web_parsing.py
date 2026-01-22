"""Web parsing utilities."""

from __future__ import annotations

import json
import logging

from argparse import ArgumentParser
from dataclasses import dataclass
from pprint import pprint
from typing import Any

from nkpylib.ml.client import call_llm
from nkbase.parser import extract_many, unify_objects

logger = logging.getLogger(__name__)

class Rule:
    """A way to define and rules for parsing pages.

    """
    def __init__(self, name: str, selector: str, attr: str|None='', sub_rules: list[Rule]|None=None, **kw):
        self.name = name
        self.selector = selector
        self._attr = attr
        self.method = 'text'  # default extraction method
        self.transforms = []  # list of transform functions
        self.sub_rules = sub_rules or []
        self.kw = kw
        self.as_list = False

    def lst(self):
        """Process each matching element separately"""
        self.as_list = True
        return self

    # jQuery-style extraction methods
    def text(self):
        """Extract text content"""
        self.method = 'text'
        return self

    def val(self):
        """Extract value (for form elements)"""
        self.method = 'val'
        return self

    def html(self):
        """Extract HTML content"""
        self.method = 'html'
        return self

    def attr(self, attr_name: str):
        """Extract attribute value"""
        self.method = 'attr'
        self._attr = attr_name
        return self

    # Transform methods
    def replace(self, old: str, new: str = ''):
        """Replace text in the extracted value"""
        self.transforms.append(lambda x: x.replace(old, new) if x else x)
        return self

    def split(self, sep: str = ' '):
        """Split the extracted value"""
        self.transforms.append(lambda x: x.split(sep) if x else [])
        return self

    def strip(self):
        """Strip whitespace"""
        self.transforms.append(lambda x: x.strip() if x else x)
        return self

    def lower(self):
        """Convert to lowercase"""
        self.transforms.append(lambda x: x.lower() if x else x)
        return self

    def make_int(self):
        """Convert to int, handling K/M suffixes"""
        def convert(s):
            if not s:
                return 0
            s = str(s).replace(',', '')
            if s.endswith('K'):
                return int(float(s[:-1]) * 1000)
            elif s.endswith('M'):
                return int(float(s[:-1]) * 1000000)
            else:
                return int(s)
        self.transforms.append(convert)
        return self

    def transform(self, func: callable):
        """Apply custom transform function"""
        self.transforms.append(func)
        return self

    def sub(self, *sub_rules):
        """Add sub-rules that process this rule's output"""
        self.sub_rules = list(sub_rules)
        return self

    @classmethod
    def int(cls, s: str, with_suffix:bool=True) -> int:
        """Parse integer from string that might end with K or M suffix"""
        if not s:
            return 0
        s = str(s).replace(',', '')
        if not with_suffix:
            return int(s)
        if s.endswith('K'):
            return int(float(s[:-1]) * 1000)
        elif s.endswith('M'):
            return int(float(s[:-1]) * 1000000)
        else:
            return int(s)

    def extract_value(self, doc):
        """Extract the value using the configured method and transforms"""
        sel = doc(self.selector)
        todo = [sel]
        if self.as_list:
            todo = [doc(s) for s in sel]
            print(f'For selector {self.selector}, found {len(todo)} items: {todo}')
        ret = []
        for o in todo:
            # Extract based on method
            match self.method:
                case 'text':
                    value = o.text()
                case 'val':
                    value = o.val()
                case 'html':
                    value = o.html()
                case 'attr':
                    value = o.attr(self._attr)
                case _:
                    value = o.text()

            # Apply transforms
            for transform_func in self.transforms:
                try:
                    value = transform_func(value)
                except Exception as e:
                    logger.info(f"Error applying transform in rule {self.name}: {e}")
                    break
            if self.as_list:
                print(f'  Added value {value}')
                ret.append(value)
            else:
                return value
        return ret

    @classmethod
    def make_shortcut(cls, lst=None) -> tuple[callable, callable, list[Rule]]:
        """Makes shortcut functions that lets you easily create Rules, etc.

        Typically used like this:

            R, S, rules = Rule.make_shortcut()

        Then you can use `R` to add rules to `rules`, `S` to create sub-rules, and then when all
        are done, you can call Rule.parse_all(doc, rules) to parse a document.
        """
        if lst is None:
            lst = []

        def add_rule(name: str, selector: str, attr: str|None='', **kw):
            rule = Rule(name, selector, attr, **kw)
            lst.append(rule)  # Add to main list
            return rule

        def sub_rule(name: str, selector: str = '', attr: str|None='', **kw):
            rule = Rule(name, selector, attr, **kw)
            # Don't add to main list
            return rule

        return add_rule, sub_rule, lst

    @classmethod
    def parse_all(cls, doc, rules: list[Rule]) -> dict[str, Any]:
        """Parses the given `doc` using the given `rules`"""
        ret = {}
        for rule in rules:
            try:
                value = rule.extract_value(doc)
                if rule.sub_rules:
                    # Process sub-rules using the extracted value as input
                    for sub_rule in rule.sub_rules:
                        try:
                            # Apply sub-rule transforms to the parent value
                            sub_value = value
                            for transform_func in sub_rule.transforms:
                                sub_value = transform_func(sub_value)
                            ret[sub_rule.name] = sub_value
                        except Exception as e:
                            logger.info(f"Error processing sub-rule {sub_rule.name}: {e}")
                            ret[sub_rule.name] = None
                else:
                    ret[rule.name] = value
            except Exception as e:
                logger.info(f"Error parsing rule {rule.name} with selector {rule.selector}: {e}")
        return ret


@dataclass
class FieldDefinition:
    """Defines a field that should be extracted from web pages"""
    name: str
    description: str
    data_type: str  # 'str', 'int', 'list', etc.
    example_values: list[str]
    required: bool = True


@dataclass
class GroundTruthSample:
    """A sample page with its extracted field values"""
    url: str
    html: str
    fields: dict[str, Any]
    confidence: float = 1.0  # How confident we are in this ground truth


class AutomaticParser:
    """A class that tries to automatically build parsers for webpages using LLMs.

    The way this works is that the input is a list of raw web pages, all from the same site or of
    the same type. I then initially ask an LLM to extract all relevant fields from each page, and
    this becomes the "ground truth" (perhaps with some manual cleaning). Once I have a list of
    fields, I ask it to generate rules to extract those fields from the page. I would provide it
    this entire Python file, or at least the `Rule` class portion of it, so it knows what it's
    working with. (And also perhaps some examples.)

    I would then execute the code from the LLM on each of the examples and record the outputs and
    any errors that occur. I would then iterate on this process by sending these outputs and errors
    to the LLM and asking it to fix the parser, i.e. the set of rules that it's created.
    """
    def __init__(self, model: str='code', examples: list[str]|None = None):
        """Initialize the AutomaticParser.

        - model: llm model to use
        - existing_parser_examples: list of existing parser code to use as examples
        """
        self.model = model
        self.examples = examples

    def run(self, substr: str):
        paths = extract_many(substr=substr, output_prefix='manual')
        #unified = unify_objects(substr, output_prefix='unified')
        pprint(unified)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    parser = ArgumentParser(description="Automatic Web Page Parser")
    parser.add_argument('substr', type=str, help="Substring to match files")
    args = parser.parse_args()
    ap = AutomaticParser()
    ap.run(args.substr)
