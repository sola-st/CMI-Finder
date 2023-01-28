from data_collection.scrape_repos import search_and_clone
from data_collection.extract_statements import get_if_stmts_tups, paralel_extractor, post_process_ifs, construct_text_data
from data_collection.mining_past_bug_fixes import construct_pairs, extractor, simplify_collected_statements, construct_text_data
from data_collection.mining_past_bug_fixes import postprocess_old_new_commits, postprocess_old_new_commits_
from data_collection.collect_data import collect_data
import json
import os

def test_search_and_clone():
    goal = 10
    auth = ("islem-esi", "Bearer TOKEN")
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
        
def test_construct_pairs():
    repo_url = "https://github.com/sola-st/DynaPyt"
    pairs = construct_pairs(repo_url)
    with open("dynapyt_pairs.json", "w") as dpj:
        json.dump(pairs, dpj)

def test_extractor():
    with open("dynapyt_pairs.json") as dpj:
        pairs_list = json.load(dpj)
    
    old_new_pairs = extractor(pairs_list)

    with open("dynapyt_old_new_pairs.json", "w") as don:
        json.dump(str(old_new_pairs), don)

def test_simplify_collected_statements():
    with open("dynapyt_pairs.json") as dpj:
        pairs_list = json.load(dpj)
    old_new_pairs = extractor(pairs_list)
    simple_ifs_stmts = simplify_collected_statements(old_new_pairs)

def test_construct_text_data():
    with open("dynapyt_pairs.json") as dpj:
        pairs_list = json.load(dpj)
        old_new_pairs = extractor(pairs_list)
        simple_ifs_stmts = simplify_collected_statements(old_new_pairs)
        repo_hash = list(simple_ifs_stmts.keys())[0]
        file_name = list(simple_ifs_stmts[repo_hash].keys())[0]
        text_data = construct_text_data(simple_ifs_stmts[repo_hash][file_name]["old"])
        with open("text_dynapyt.json", "w") as tdj:
            json.dump(text_data, tdj)

def test_postprocess_old_new_commits():
    with open("dynapyt_pairs.json") as dpj:
        pairs_list = json.load(dpj)
    old_new_pairs = extractor(pairs_list)
    commit_old_new = postprocess_old_new_commits(old_new_pairs)

def test_postprocess_old_new_commits_():
    with open("dynapyt_pairs.json") as dpj:
        pairs_list = json.load(dpj)
    old_new_pairs = extractor(pairs_list)
    commit_old_new = postprocess_old_new_commits(old_new_pairs)
    commit_old_new_ = postprocess_old_new_commits_(commit_old_new)

def test_collect_data():
    pass
