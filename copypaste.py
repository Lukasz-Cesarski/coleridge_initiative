import glob
import json
import re
import os
import shutil

import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from os.path import join
from utils import r_jsonl, clean_text, w_jsonl, JSONLReader, r_json
from datetime import datetime
from copy import deepcopy

LABEL_DT = "DT"  # dataset_title
MP_PROCESSES = 6
MP_PROCESSES = 6

def parse_article(sort_idx, path, df_train):
    ### ARTICLES ###
    article = {}
    article["path"] = path
    paper_idx = os.path.split(path)[-1][:-5]
    article["idx"] = paper_idx
    paper = r_json(path)
    # we assume single whitespace in further processing
    paper = [{k: " ".join(v.split()) for k, v in part.items()} for part in paper]
    article["paper"] = paper
    clean_paper = [
        {
            "section_title": part["section_title"],
            "text": clean_text(part["text"]),
            "matches": [],
        } for part in paper
    ]
    article["paper_clean"] = clean_paper

    ### WORDMAP ###
    for raw_part, clean_part in zip(paper, clean_paper):
        raw_txt = raw_part["text"]
        clean_txt = clean_part["text"]
        raw_tokens = raw_txt.split()
        clean_tokens = [clean_text(t) for t in raw_tokens]
        wordmap = []
        filter_tokens = []
        for idx, t in enumerate(clean_tokens):
            if not re.fullmatch(r"^\s*$", t):
                filter_tokens.append(t)
                wordmap.extend([idx for _ in range(len(t.split(" ")))])
        if wordmap:
            wordmap.append(wordmap[-1])
        # checks
        processed_txt = " ".join(filter_tokens)
        assert clean_txt == processed_txt
        clean_part["wordmap"] = wordmap

    ### INSTANCES ###
    instances_df = df_train.query("Id == @paper_idx")
    instances = [instance_row.to_dict() for _, instance_row in instances_df.iterrows()]
    article["instances"] = instances

    ### MATCHES ###
    for instance in instances:
        pattern = instance["dataset_label"]
        pattern = r"\b" + clean_text(pattern) + r"\b"
        for part in clean_paper:
            txt = part["text"]
            matches = list(re.finditer(pattern, txt))
            if matches:
                part["matches"].extend(
                    [{"start": m.start(), "end": m.end(), "label": LABEL_DT} for m in matches])

    ### ALIGNMENT ###
    for raw_part, clean_part in zip(paper, clean_paper):
        clean_matches = clean_part["matches"]
        raw_matches = []
        clean_txt = clean_part["text"]
        wordmap = clean_part["wordmap"]
        raw_txt = raw_part["text"]
        clean_cumsum = np.cumsum([0] + [len(t) + 1 for t in clean_txt.split()])
        list_clean_cumsum = clean_cumsum.tolist()
        raw_cumsum = np.cumsum([0] + [len(t) + 1 for t in raw_txt.split()])
        list_raw_cumsum = raw_cumsum.tolist()

        if clean_matches:
            # clean_matches = deoverlap(clean_matches)
            for clean_match in clean_matches:
                clean_start_char = clean_match["start"]
                clean_end_char = clean_match["end"]
                clean_match_txt = clean_txt[clean_start_char:clean_end_char]
                clean_start_word = list_clean_cumsum.index(clean_start_char)
                clean_end_word = list_clean_cumsum.index(clean_end_char + 1)
                clean_match["start_word"] = clean_start_word
                clean_match["end_word"] = clean_end_word
                assert clean_match_txt == " ".join(clean_txt.split()[clean_start_word:clean_end_word])
                raw_start_word = wordmap[clean_start_word]
                raw_end_word = wordmap[clean_end_word]
                if raw_end_word == wordmap[clean_end_word - 1]:
                    raw_end_word += 1
                if raw_start_word == raw_end_word:
                    raw_end_word += 1
                raw_start_char = list_raw_cumsum[raw_start_word]
                raw_end_char = list_raw_cumsum[raw_end_word] - 1
                raw_match_txt = raw_txt[raw_start_char:raw_end_char]
                assert raw_match_txt == " ".join(raw_txt.split()[raw_start_word:raw_end_word])
                assert clean_match_txt in clean_text(raw_match_txt)
                raw_match = {
                    "start": raw_start_char,
                    "end": raw_end_char,
                    "start_word": raw_start_word,
                    "end_word": raw_end_word,
                    "label": LABEL_DT,
                }
                raw_matches.append(raw_match)
        raw_part["matches"] = raw_matches
    return sort_idx, article


def preprocess_data(data_dir, save_path):
    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    train_files = glob.glob(join(data_dir, "train/*.json"))
    test_files = glob.glob(join(data_dir, "test/*.json"))
    train_file = join(data_dir, "train.csv")
    id_to_path = {
        os.path.split(path)[-1][:-5]: path for path in train_files
    }
    assert len(id_to_path) == len(train_files)
    assert os.path.isfile(train_file)
    print("Train Atricles found", len(train_files))
    print("Test Articles found", len(test_files))
    df_train = pd.read_csv(train_file)
    start = datetime.now()
    articles = []
    train_files = train_files[:100]
    BATCH_SIZE = 2000
    with open(save_path, "w") as fw:
        for tfs in tqdm(range(0, len(train_files), BATCH_SIZE)):
            train_files_batch = train_files[tfs:tfs+BATCH_SIZE]
            pool = mp.Pool(processes=MP_PROCESSES)
            processes = [pool.apply_async(parse_article, args=(sort_idx, path, df_train)) for sort_idx, path in enumerate(train_files_batch)]
            output_tuples = [p.get() for p in processes]
            pool.close()
            for _, art in sorted(output_tuples, key=lambda x: x[0]):
                json.dump(art, fw, ensure_ascii=False)
                fw.write('\n')
    print(f"Processing time {datetime.now() - start}")
    return articles


if __name__ == "__main__":
    DATA_DIR = "data"
    PROCESSED_PATH = "output/01_preprocessed_data.jsonl"
    FILTER_PATH = "output/02_filter_data.jsonl"
    if not os.path.isfile(PROCESSED_PATH):
        preprocess_data(DATA_DIR, PROCESSED_PATH)
    else:
        size = os.path.getsize(PROCESSED_PATH)/1000**2
        print(f"Data found. Size: {size:.3f} MB")

    print("Filtering...")
    start = datetime.now()
    if not os.path.isfile(FILTER_PATH):
        with open(PROCESSED_PATH) as fr, open(FILTER_PATH, "w") as fw:
            for article in JSONLReader(fr):
                article_filter = deepcopy(article)
                article_filter["paper"] = [part for part in article["paper"] if part["matches"]]
                article_filter["paper_clean"] = [part for part in article["paper_clean"] if part["matches"]]
                assert len(article_filter["paper"]) == len(article_filter["paper_clean"])
                json.dump(article_filter, fw, ensure_ascii=False)
                fw.write('\n')
    print(f"Filtering finished {datetime.now() - start}")
