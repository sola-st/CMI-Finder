import argparse
import os
from data_collection.extract_statements import paralel_extractor, post_process_ifs, ifstmts_to_json
from data_collection.scrape_repos import clone_projects, search_and_clone
from getpass import getpass


parser = argparse.ArgumentParser()

parser.add_argument(
    "--strategy", help="""The strategy can be one of:
        random: scrape random repos from github
        list: scrape a given list of repositories, the list is stored in a text file, each line is a full name of a repo in the format repo-owner/repo-name
        folder: collect data from a folder of repositories,
        query: scrape from github but with given criterions""", default="random")

parser.add_argument(
    "--size", help="The number of repositories to scrape")

parser.add_argument(
    "--strategy_arg", help="""
    This is a complementary input to the strategy, it works as follows:
        when strategy == random: this argument is not required and has no effect
        when strategy == list: this argument should contain the path to the txt file containing the list of repositories
        when strategy == folder: this argument should contain the path to the folder containing the repositories
        when strategy == query: this argument should contain the selection criterion in the form of github https api scraping criteria""")

parser.add_argument(
    "--output", help="path to folder where the collected data should be saved", required=True)

parser.add_argument(
    "-n", help="number of cpus", default=1)

def collect_data(projects_list, base_dir, n_cpus):
    if_stmts = paralel_extractor(projects_list, base_dir, n_cpus=n_cpus)
    ifs_by_project = post_process_ifs(if_stmts)
    return ifs_by_project

if __name__ == "__main__":
    args = parser.parse_args()
    strategy = args.strategy

    if strategy == "random":
        print("To scrape repositories from github, you need a user name and a corresponding github token")
        user_name = input("GitHub user name:")
        token = getpass("GitHub token:")
        goal = int(args.size)
        base_dir = args.output
        search_and_clone(goal=goal, auth=(user_name, token), base_dir=base_dir)
        #projects = [d for d in os.listdir(base_dir) if not d.endswith(".json")]
        #ifs_by_project = collect_data(projects, base_dir, args.n)
        ## save to json
        #ifstmts_to_json(ifs_by_project, save_name=os.path.join(args.output, "if_stmts_by_project_random"))
        print("Collection finished")
        #print("Results were saved to:", os.path.join(args.output, "if_stmts_by_project_random.json"))

    elif strategy == "list":
        if args.strategy_arg is None:
            raise ValueError("--strategy_arg should be provided in the command line (path to file containing repos list)")
        if not os.path.exists(args.strategy_arg):
            print("The path to the file you provided does not exist")
            print("The path of the entry point is at:", os.getcwd())
            print("Consider giving a path relative to the current path or an absolute paht of the folder")
            raise Exception
        with open(args.strategy_arg) as lstf:
            projects = lstf.read().splitlines()
        os.mkdir(os.path.join(args.output, "cloned_repo_folder_123546865"))
        clone_projects(projects, len(projects), base_dir=os.path.join(args.output, "cloned_repo_folder_123546865"))
        #projects_list = os.listdir(os.path.join(args.output, "cloned_repo_folder_123546865"))
        #ifs_by_project = collect_data(projects_list, os.path.join(args.output, "cloned_repo_folder_123546865"), args.n)
        #ifstmts_to_json(ifs_by_project, save_name=os.path.join(args.output, "if_stmts_by_project_list"))
        print("Collection finished")
        #print("Results were saved to:", os.path.join(args.output, "if_stmts_by_project_list.json"))
    elif strategy == "folder":
        if not os.path.isdir(args.strategy_arg):
            print("The folder you provided does not exist")
            print("The path of the entry point is at:", os.getcwd())
            print("Consider giving a path relative to the current path or an absolute paht of the folder")
            raise Exception
        projects = os.listdir(args.strategy_arg)
        ifs_by_project = collect_data(projects, args.strategy_arg, args.n)
        ## save to json
        ifstmts_to_json(ifs_by_project, save_name=os.path.join(args.output, "if_stmts_by_project_folder"))
        print("Data collection finished")
        print("Results were saved to:", os.path.join(args.output, "if_stmts_by_project_folder.json"))
    elif strategy == "query":
        print("To scrape repositories from github, you need a user name and a corresponding github token")
        user_name = input("GitHub user name:")
        token = input("GitHub token:")
        goal = args["size"]
        base_dir = args.output
        raise NotImplemented("not yet implemented")
