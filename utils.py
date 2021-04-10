import json
import re
import os
from pprint import pprint
import pandas as pd
from typing import Optional, Callable, Dict, Iterable, Any, Generator
from datetime import datetime


def r_json(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def w_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def r_jsonl(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def w_jsonl(data: list, file_path):
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def jaccard_similarity(s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union


def describe_article(article):
    instances = article["instances"]
    print(f"INSTANCES: {len(instances)}")
    pprint(instances)
    print("\n\n")
    print("MATCHES")
    docs = []
    for raw_part, clean_part in zip(article["paper"], article["paper_clean"]):
        clean_matches = clean_part["matches"]
        raw_matches = raw_part["matches"]
        assert bool(clean_matches) == bool(raw_matches)
        clean_txt = clean_part["text"]
        raw_txt = raw_part["text"]
        clean_tokens = clean_txt.split()
        raw_tokens = raw_txt.split()
        # wordmap = clean_part["wordmap"]
        if clean_matches:
            docs.append(
                {"text": clean_txt,
                 "ents": clean_matches,
                 "title": clean_part["section_title"]})
            docs.append(
                {"text": raw_txt,
                 "ents": raw_matches,
                 "title": raw_part["section_title"]})
            for c_match, r_match in zip(clean_matches, raw_matches):
                c_text = clean_txt[c_match["start"]:c_match["end"]]
                c_tokens = clean_tokens[c_match["start_word"]:c_match["end_word"]]
                r_text = raw_txt[r_match["start"]:r_match["end"]]
                r_tokens = raw_tokens[r_match["start_word"]:r_match["end_word"]]
                print("c_text   ", c_text)
                print("c_tokens ", c_tokens)
                print("r_text   ", r_text)
                print("r_tokens ", r_tokens)
                print()


class JSONLReader:
    """Example data reader that reads jsonl entries"""

    def __init__(self, data: Iterable[Any], *args: Any) -> None:
        self.data = data
        self.args = args

    def _process(self, entry: str) -> Dict[str, Any]:
        return json.loads(entry)

    def __iter__(self) -> Generator[Any, None, None]:
        for entry in self.data:
            yield self._process(entry)


def deoverlap(matches):
    deover_matches = []
    for this_match in matches:
        for match_i in matches:
            if this_match == match_i:
                continue  # self
            if this_match["start"] >= match_i["end"]:
                continue  # located left
            if this_match["end"] <= match_i["start"]:
                continue  # loacted right
            start_diff = this_match["start"] - match_i["start"]
            end_diff = this_match["end"] - match_i["end"]
            if start_diff >= 0 and end_diff <= 0:
                break  # hidden inside other match
            if start_diff * end_diff > 0:
                raise ValueError(f"{this_match}{match_i}:{start_diff}:{end_diff}")
        else:
            deover_matches.append(this_match)
    return deover_matches


# def find_matches_for_instance(instance: pd.Series, clean_function: Optional[Callable] = None):
#     """
#     1. Find corresponding article
#     2. Localize Dataset Name Instances (dataset_label)
#     3. Prepare data for spacy.displacy
#     """
#     article = r_json(id_to_path[instance["Id"]])
#     docs = []
#     pattern = instance["dataset_label"]
#     if clean_function:
#         pattern = clean_function(pattern)
#     for part in article:
#         title = part["section_title"]
#         txt = part["text"]
#         if clean_function:
#             txt = clean_function(txt)
#         matches = list(re.finditer(pattern, txt))
#         if matches:
#             docs.append({
#                 "text": txt,
#                 "ents": [{"start": m.start(), "end": m.end(), "label": LABEL_DT} for m in matches],
#                 "title": title,
#             })
#     return docs
#
#
# def summarize_cleaning_policy(clean_function=None):
#     start = datetime.now()
#     matched_ids = []
#     missing_ids = []
#     for idx in range(len(df_train)):
#         docs = find_matches_for_instance(df_train.iloc[idx], clean_function=clean_function)
#         if docs:
#             matched_ids.append(idx)
#         else:
#             missing_ids.append(idx)
#     print(f"Processing time {datetime.now() - start}")
#     print(f"Matched ids: {len(matched_ids)}")
#     print(f"Missing ids: {len(missing_ids)}")
#
#     ### error analysis
#     if not missing_ids:
#         print("Every dataset instance matched! GREAT!")
#     else:
#         for idx in range(min(len(missing_ids), 5)):
#             instance = df_train.iloc[missing_ids[idx]]
#             pattern = instance["dataset_label"]
#             article = r_json(id_to_path[instance["Id"]])
#             found = []
#             pat_len = len(pattern)
#             stride = 5
#             fuzz_threshold = 90  # range from 0-100, 100 means match
#             for part in article:
#                 txt = part["text"]
#                 for start_idx in range(0, len(txt), stride):
#                     chunk = txt[start_idx:start_idx + pat_len + stride]
#                     if fuzz.partial_ratio(pattern, chunk) > fuzz_threshold:
#                         found.append(chunk)
#             print(f"Pattern = {pattern}")
#             print(f"Instances = {found}")
#             print()
#     return matched_ids, missing_ids



