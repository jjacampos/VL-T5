import argparse
import json
import random
import re


def main(args):

    to_avoid = ["first", "second", "third", "fourth", "this", "that", "share it", "was it", "here"]
    data = json.load(open(args.data_path, 'r'))
    just_text_examples = []
    mm_examples = []
    for elem in data:
        sentence_splitted = elem['predict'].split('<USER>')
        if elem['type']=='RESPONSE' or len(sentence_splitted)<=2 or len(re.findall('(\d{5,})', elem['target'])) == 0:
            continue
        last_user_utterance = sentence_splitted[-1]
        found = False
        for special_token in to_avoid:
            if special_token in last_user_utterance.lower():
                just_text_examples.append(elem)
                found = True
        if not found:
            mm_examples.append(elem)

    print(f'Just text examples: {len(just_text_examples)}\t{random.sample(just_text_examples, 1)[0]}')
    print(f"MM examples: {len(mm_examples)}\t{random.sample(mm_examples, 1)[0]}")

    output_dir = f"{args.data_path.split('.json')[0]}_just_mm.json"
    with open(output_dir, 'w', encoding='utf-8') as output_file:
        print(len(mm_examples))
        json.dump(mm_examples, output_file)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the dataset",
    )
    
    args = parser.parse_args()

    main(args)