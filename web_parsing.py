"""Web parsing utilities."""

from __future__ import annotations

import json
import logging

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

import pyquery

from nkpylib.ml.client import call_llm
from nkpylib.ml.llm_utils import load_llm_json
from nkbase.parser import extract_many, unify_objects

logger = logging.getLogger(__name__)

class Rule:
    """A way to define rules for parsing web pages using CSS selectors and transformations.

    This class provides a fluent interface for building web scraping rules. Each rule specifies:
    - A field name to extract
    - A CSS selector to find elements
    - An extraction method (text, attribute, etc.)
    - Optional transformations to apply to the extracted data
    - Optional sub-rules for further processing

    Basic usage:
        Rule('title', 'h1').text().strip()
        Rule('price', '.price').text().replace('$', '').make_int()
        Rule('links', 'a').lst().attr('href')

    The rule can be configured with method chaining:
    - Extraction methods: `.text()`, `.attr(name)`, `.val()`, `.html()`
    - List processing: `.lst()` to process each matching element separately
    - Transformations: `.strip()`, `.replace()`, `.split()`, `.lower()`, `.make_int()`
    - Custom transforms: `.transform(func)`
    - Sub-rules: `.sub(rule1, rule2, ...)` for nested processing

    Args:
    - name: Field name for the extracted data
    - selector: CSS selector to find elements
    - attr: Attribute name (used with `.attr()` method)
    - sub_rules: List of `Rule` objects for nested processing
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
        #pprint(unified)


def generate_rules(html_file: str, prompt: str, model: str = 'html') -> dict[str, Any]:
    """Generate parsing rules using LLM"""
    html_content = Path(html_file).read_text()
    # Create a simplified version of HTML for the LLM
    doc = pyquery.PyQuery(html_content)
    # Remove script and style tags
    doc.remove('script, style')
    simplified_html = doc.html()
    full_prompt = f"""
Given this HTML content, generate Python code using the Rule class to extract data fields.

HTML:
{simplified_html[:5000]}...

User request: {prompt}

Please generate a list of Rule objects that can extract the requested data. Use the Rule class methods like .text(), .attr(), .lst(), etc. Return only the Python code that creates the rules list.

Example format:
```python
rules = [
    Rule('title', 'h1').text().strip(),
    Rule('price', '.price').text().replace('$', '').make_int(),
    Rule('tags', '.tag').lst().text()
]
```
"""
    response = call_llm.single(full_prompt, model=model)
    # Extract the Python code from the response
    if '```python' in response:
        code_start = response.find('```python') + 9
        code_end = response.find('```', code_start)
        code = response[code_start:code_end].strip()
    else:
        code = response.strip()
    return dict(code=code, full_response=response)

def test_rules(html_file: str, rules_file: str) -> dict[str, Any]:
    """Test existing rules on an HTML file"""
    html_content = Path(html_file).read_text()
    rules_content = Path(rules_file).read_text()
    doc = pyquery.PyQuery(html_content)
    # Execute the rules code to get the rules list
    local_vars = dict(Rule=Rule)
    exec(rules_content, {}, local_vars)
    rules = local_vars.get('rules', [])
    if not rules:
        return dict(error="No 'rules' variable found in rules file")
    try:
        results = Rule.parse_all(doc, rules)
        return dict(success=True, results=results)
    except Exception as e:
        return dict(error=f"Error running rules: {e}")

def main():
    parser = ArgumentParser(description="Web Page Parser - Generate or test parsing rules")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    # Generate rules command
    gen_parser = subparsers.add_parser('generate', help='Generate parsing rules using LLM')
    gen_parser.add_argument('html_file', help='HTML file to analyze')
    gen_parser.add_argument('prompt', help='Description of what data to extract')
    gen_parser.add_argument('--model', default='html', help='LLM model to use')
    gen_parser.add_argument('--output', '-o', help='Output JSON file for rules')
    # Test rules command
    test_parser = subparsers.add_parser('test', help='Test existing rules on HTML file')
    test_parser.add_argument('html_file', help='HTML file to parse')
    test_parser.add_argument('rules_file', help='JSON file containing parsing rules')
    # parse args and run
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    match args.command:
        case 'generate':
            result = generate_rules(args.html_file, args.prompt, args.model)
            if args.output:
                Path(args.output).write_text(json.dumps(result, indent=2))
                print(f"Rules saved to {args.output}")
            else:
                print("Generated rules:")
                print(result['code'])
        case 'test':
            result = test_rules(args.html_file, args.rules_file)
            if result.get('success'):
                print("Parsing results:")
                pprint(result['results'])
            else:
                print(f"Error: {result['error']}")

def old_auto_main():
    parser = ArgumentParser(description="Automatic Web Page Parser")
    parser.add_argument('substr', type=str, help="Substring to match files")
    args = parser.parse_args()
    ap = AutomaticParser()
    ap.run(args.substr)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    main()
