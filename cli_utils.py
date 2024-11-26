"""Command-line tools"""

from __future__ import annotations

import string

from typing import Any

# an action is a name and a function
Action = tuple[str, callable]

def perform_actions_on_items(items: list[Any], actions: dict[str, Action]):
    """
    Perform actions on a list of items based on user input.

    :param items: List of items to perform actions on.
    :param actions: Dictionary mapping action letters to (action_name, action_func).
    """
    item_labels = string.digits + string.ascii_lowercase + string.ascii_uppercase
    item_map = {label: item for label, item in zip(item_labels, items)}

    if len(items) > len(item_labels):
        print("Error: Too many items to enumerate with single characters.")
        return

    item_list = ', '.join(f"{label}:{item}" for label, item in item_map.items())
    action_list = ', '.join(f"{name}({letter})" for letter, (name, _) in actions.items())

    while True:
        user_input = input(f"Items: {item_list} | Actions: {action_list} | Enter actions: ").strip()
        try:
            for action_spec in user_input.split(','):
                action_letter, item_spec = action_spec.split(':')
                if action_letter not in actions:
                    print(f"Error: Invalid action '{action_letter}'.")
                    continue
                action_name, action_func = actions[action_letter]
                selected_items = parse_item_spec(item_spec, item_map)
                for item in selected_items:
                    action_func(item)
        except Exception as e:
            print(f"Error: {e}")

def parse_item_spec(item_spec, item_map):
    """
    Parse the item specification and return the corresponding items.

    :param item_spec: String specifying items, e.g., '1-3a'.
    :param item_map: Dictionary mapping item labels to items.
    :return: List of items corresponding to the specification.
    """
    items = []
    i = 0
    while i < len(item_spec):
        if i + 2 < len(item_spec) and item_spec[i+1] == '-':
            start, end = item_spec[i], item_spec[i+2]
            if start not in item_map or end not in item_map:
                raise ValueError(f"Invalid range '{start}-{end}'.")
            start_idx = list(item_map.keys()).index(start)
            end_idx = list(item_map.keys()).index(end)
            items.extend(item_map[label] for label in list(item_map.keys())[start_idx:end_idx+1])
            i += 3
        else:
            if item_spec[i] not in item_map:
                raise ValueError(f"Invalid item '{item_spec[i]}'.")
            items.append(item_map[item_spec[i]])
            i += 1
    return items
