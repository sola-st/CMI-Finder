from data_collection.scrape_repos import search_and_clone
from data_collection.extract_statements import get_if_stmts_tups, paralel_extractor, post_process_ifs, construct_text_data
import json
import os

def test_search_and_clone():
    goal = 10
    auth = ("islem-esi", "Bearer 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")
    base_dir = "cloned_repos"
    search_and_clone(goal=goal, auth=auth, base_dir=base_dir)

def test_get_if_stmts_tups():
    concerned_projects = os.listdir("cloned_repos")
    start = 0
    end = 5
    if_stmts = get_if_stmts_tups(concerned_projects, start, end)
    print(if_stmts.keys())

def test_paralel_extractor():
    n_cpus=4
    concerned_projects = os.listdir("cloned_repos")
    paralel_extractor(concerned_projects, n_cpus=n_cpus)

def test_post_process_ifs():
    n_cpus=4
    concerned_projects = os.listdir("cloned_repos")
    ifstmts_by_repo = paralel_extractor(concerned_projects, n_cpus=n_cpus)
    post_ifs = post_process_ifs(ifstmts_by_repo)
    with open("post_ifs_file.json", 'w') as pif:
        json.dump(post_ifs, pif)

def test_construct_text_data():
    with open("post_ifs_file.json") as pif:
        if_stmts = json.load(pif)

    ifs_list = []
    for key in if_stmts:
        ifs_list.extend(if_stmts[key])

    text_data = construct_text_data(ifs_list)
    with open("text_data_ifs.json", "w") as tdi:
        json.dump(text_data, tdi)
        