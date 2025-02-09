#! /usr/bin/env python
"""
Create API and MM-DST result JSONS from model result file.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import copy
import json
import ast
import re


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(r"([\w:?.?]*) *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>")
    # My update for avoiding the issue in the that I wrote in the paper
    # slot_regex = re.compile(r"([A-Za-z0-9_.-:]*) *= *(\[(.*)\]|[^,]*)")
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*) *= *(\[[^]]*\]|[^,]*)")

    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    to_parse = to_parse.strip()
    # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
    # Originally slots is a dict if not then is readed as a string

    for dialog_act in dialog_act_regex.finditer(to_parse):
        """
            d = {
            "act": dialog_act.group(1),
            "slots": [],
            "request_slots": [],
            "memories": [],
        }

        for slot in slot_regex.finditer(dialog_act.group(2)):
            d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])
        """
        d = {
            "act": dialog_act.group(1),
            "slots": {},
            "request_slots": [],
            "memories": [],
        }

        for slot in slot_regex.finditer(dialog_act.group(2)):
            try:
                if "[" in slot.group(2):
                    slot_list = ast.literal_eval(slot.group(2).strip())
                else:
                    slot_list = [slot.group(2).strip()]
                d["slots"][slot.group(1).strip()]= slot_list
            # Still problem when the names don't have '' on them
            except:
                print('If happens many times improve parsing')
                d["slots"][slot.group(1).strip()] = []
            
        for request_slot in request_regex.finditer(dialog_act.group(3)):
            d["request_slots"].append(request_slot.group(1).strip())
        for object_id in object_regex.finditer(dialog_act.group(4)):
            d["memories"].append(object_id.group(1).strip())
        if d != {}:
            belief.append(d)
    return belief


def create_result_jsons(results, test_data):
    """Creates two JSON files from results.

    Args:
        results: List of generated results from the model.
        test_data: Raw JSON test file.

    Returns:
        response_results: Dict containing response results
        dst_results: Dict containing DST results
    """
    dst_results = copy.deepcopy(test_data)
    response_results = collections.defaultdict(list)
    dst_pool = {}
    for instance in results:
        dialog_id = instance["dialog_id"]
        turn_id = instance["turn_id"]
        if instance["type"] == "API":
            index = (dialog_id, turn_id)
            dst_pool[index] = instance
        else:
            if dialog_id not in response_results:
                response_results[dialog_id] = {
                    "dialog_id": dialog_id,
                    "predictions": [],
                }
            response_results[dialog_id]["predictions"].append(
                {
                    "turn_id": turn_id,
                    "response": instance["model_prediction"],
                }
            )
    num_missing = 0
    num_present = 0
    for dialog_datum in dst_results["dialogue_data"]:
        del dialog_datum["mentioned_memory_ids"]
        del dialog_datum["memory_graph_id"]
        dialog_id = dialog_datum["dialogue_idx"]
        for datum in dialog_datum["dialogue"]:
            turn_id = datum["turn_idx"]
            index = (dialog_id, turn_id)
            if index in dst_pool:
                model_pred_datum = dst_pool[index]
                model_pred = model_pred_datum["model_prediction"].strip(" ")
                parsed_result = parse_flattened_result(model_pred)
                datum["transcript_annotated"] = parsed_result
                num_present += 1
            else:
                del datum["transcript_annotated"]
                print(f"Missing! -- {index}")
                num_missing += 1
    print(f"Missing: {num_missing} Present: {num_present}")
    return list(response_results.values()), dst_results


def main(args):
    with open(args["memory_test_json"], "r") as file_id:
        test_data = json.load(file_id)
    with open(args["model_output_json"], "r") as file_id:
        results = json.load(file_id)
    response_results, dst_results = create_result_jsons(results, test_data)

    # Save the results.
    response_results_path = args["model_output_json"].replace(
        ".json", "_response_results.json"
    )
    with open(response_results_path, "w") as file_id:
        json.dump(response_results, file_id)
    dst_results_path = args["model_output_json"].replace(".json", "_dst_results.json")
    with open(dst_results_path, "w") as file_id:
        json.dump(dst_results, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory_test_json",
        required=True,
        help="JSON file for test data",
    )
    parser.add_argument(
        "--model_output_json", required=True, help="JSON file with model outputs"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)