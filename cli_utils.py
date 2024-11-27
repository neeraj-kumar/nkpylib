"""Command-line tools"""

from __future__ import annotations

import random
import re
import readline
import string

from typing import Callable, TypeVar, Iterable

# Type variable for input items
InputT = TypeVar('InputT')

# an action is a name and a function
Action = tuple[str, Callable[[list[InputT]], Iterable[InputT] | None]]

def cli_item_action_loop(items: list[InputT],
                         actions: dict[str, Action],
                         exclusive: bool = True,
                         print_func: Callable[[InputT], str] = str,
                         max_items: int = 0) -> None:
    """
    Perform actions on a list of items based on user input in a loop until all items are done.

    :param items: List of items to perform actions on.
    :param actions: Dictionary mapping action letters to (action_name, action_func).
    :param exclusive: If True, prevents multiple different actions on the same item.
    :param print_func: Function to convert each item to a string for display.
    """
    item_labels = string.digits + string.ascii_lowercase + string.ascii_uppercase
    max_items_per_batch = min(max_items if max_items > 0 else len(item_labels), len(item_labels))
    total_items = len(items)
    start_index = 0
    batch_num = 1

    while start_index < total_items:
        end_index = min(start_index + max_items_per_batch, total_items)
        current_batch = items[start_index:end_index]
        item_map = {label: item for label, item in zip(item_labels, current_batch)}
        item_done = {item: False for item in current_batch}

        action_list = ', '.join(f"{name}({letter})" for letter, (name, _) in actions.items())
        while not all(item_done.values()):
            remaining_count = sum(not done for done in item_done.values())
            print(f"\n{remaining_count} items remaining in batch {batch_num} of {total_items//max_items_per_batch+1}")
            for label, item in item_map.items():
                if not item_done[item]:
                    print(f"{label}: {print_func(item)}")
            print()
            user_input = input(f"Actions: {action_list} > ").strip()
            try:
                done_items = parse_user_input(user_input, actions, item_map, exclusive, item_done)
                for item in done_items:
                    item_done[item] = True
            except Exception as e:
                print(f"Error: {e}")

        start_index += max_items_per_batch
        batch_num += 1

def parse_user_input(user_input: str,
                     actions: dict[str, Action],
                     item_map: dict[str, InputT],
                     exclusive: bool = False,
                     item_done: dict[InputT, bool] = None) -> list[InputT]:
    """
    Parse and execute user input actions on items.

    :param user_input: The input string from the user specifying actions and items.
    :param actions: Dictionary mapping action letters to (action_name, action_func).
    :param item_map: Dictionary mapping item labels to items.
    :param exclusive: If True, prevents multiple different actions on the same item.
    :return: List of items that have been marked as done.
    """
    action_items_map: dict[str, set[InputT]] = {action: set() for action in actions}

    item_action_map = {item: None for item in item_map.values()}

    for action_spec in re.split(r'[,\s;]+', user_input.strip()):
        action_letter = action_spec[0]
        item_spec = action_spec[2:] if action_spec[1] == ':' else action_spec[1:]
        if action_letter not in actions:
            print(f"Error: Invalid action '{action_letter}'.")
            continue
        selected_items = [item for item in parse_item_spec(item_spec, item_map) if not item_done[item]]
        action_items_map[action_letter].update(selected_items)

    done_items: list[InputT] = []
    for action_letter, items in action_items_map.items():
        if items:
            _, action_func = actions[action_letter]
            result = action_func(list(items))
            if result is not None:
                done_items.extend(item for item in result if item in item_map.values())
            #print(f"Action '{action_letter}' done on items: {', '.join(map(str, items))}")
    return done_items

def parse_item_spec(item_spec: str, item_map: dict[str, InputT]) -> list[InputT]:
    """
    Parse the item specification and return the corresponding items.

    :param item_spec: String specifying items, e.g., '1-3a'.
    :param item_map: Dictionary mapping item labels to items.
    :return: List of items corresponding to the specification.
    """
    items: list[InputT] = []
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

def generate_test_data() -> tuple[list[str], dict[str, Action]]:
    """
    Generate a simple set of items and actions for testing.

    :return: A tuple containing a list of items and a dictionary of actions.
    """
    items = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon', 'mango']

    def print_items(items: list[str]) -> list[str]:
        done = []
        for item in items:
            print(f"Item: {item}")
            done.append(item)
        return done

    def uppercase_items(items: list[str]) -> list[str]:
        done = []
        for item in items:
            print(f"Uppercased: {item.upper()}")
            done.append(item)
        return done

    actions = {
        'p': ('print', print_items),
        'u': ('uppercase', uppercase_items),
    }

    return items, actions

def test_cli_with_random_inputs(items: list[InputT],
                                actions: dict[str, Action],
                                n: int = 10,
                                exclusive: bool = False) -> None:
    """
    Test the CLI with randomly generated user input strings.

    :param items: List of items to perform actions on.
    :param actions: Dictionary mapping action letters to (action_name, action_func).
    :param n: Number of random inputs to generate for testing.
    """
    item_labels = string.digits + string.ascii_lowercase + string.ascii_uppercase
    valid_labels = item_labels[:len(items)]
    action_letters = list(actions.keys())

    for _ in range(n):
        # Generate a random input string
        num_actions = random.randint(1, 3)
        input_parts = []
        for _ in range(num_actions):
            action = random.choice(action_letters)
            item_spec = ''.join(random.choice(valid_labels) for _ in range(random.randint(1, 3)))
            input_parts.append(f"{action}:{item_spec}")
        user_input = ','.join(input_parts)

        print(f"Testing with input: {user_input}")
        try:
            item_map = {label: item for label, item in zip(valid_labels, items)}
            parse_user_input(user_input, actions, item_map, exclusive)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    # Generate test data
    items, actions = generate_test_data()

    # Run tests with random inputs
    #test_cli_with_random_inputs(items, actions)

    # Run the CLI loop with user interaction
    cli_item_action_loop(items, actions, exclusive=True)
