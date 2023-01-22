import libcst as cst
from data_collection.libcst_utils import FindIf, get_simple_ifs, remove_else
from multiprocessing import Pool
import json
import os

def get_if_stmts_tups(concerned_projects, start, end, base_dir='./selected_projects/'):
    """
    get_if_stmts_tups(concerned_projects: List[int], start: int, end: int, base_dir: str = './selected_projects/') -> Dict[int, List[Tuple[List[str], List[str]]]]

    This function takes a list of repository ids, a start index and an end index, and a directory path where the repositories are saved.
    It parses the python files of the repositories in the specified range and returns a dictionary with keys as repository ids and values as a list of tuples representing if statements.
    Each tuple contains 4 elements : the if statement with its condition, the if statement in CST format, the else statement in CST format and the else if statement in CST format.

    Parameters:
        concerned_projects (List[int]): The list of repository ids that should be processed.
        start (int): The start index of the repositories to be processed in the concerned_projects list.
        end (int): The end index of the repositories to be processed in the concerned_projects list.
        base_dir (str, optional): The directory where the repositories are saved. Default is './selected_projects/'

    Returns:
        Dict[int, List[Tuple[List[str], List[str]]]]: A dictionary with keys as repository ids and values as a list of tuples representing if statements.

    """
    ifstmt_by_repo = {}
    for repo in concerned_projects[start:end]:
        ifstmt_by_repo[repo] = []
        for subdir, dirs, files in os.walk(os.path.join(base_dir, str(repo))):
            for file in files:
                if file.endswith(".py"):
                    finder = FindIf()
                    try:
                        with open(os.path.join(subdir, file)) as py_file:
                            code = py_file.read()
                    except:
                        continue
                    try:
                        tree = cst.parse_module(code)
                        _ = tree.visit(finder)
                        ifstmt_by_repo[repo] += get_simple_ifs(finder)
                    except:
                        pass
    return ifstmt_by_repo


def paralel_extractor(concerned_projects, n_cpus=40):
    """
    paralel_extractor(concerned_projects: List[int], n_cpus: int = 40) -> Dict[int, List[Tuple[List[str], List[str]]]]

    This function takes a list of repository ids and a number of cpus.
    It uses the multiprocessing library to run the get_if_stmts_tups function in parallel on different cores, by dividing the list of repositories into n_cpus parts.
    It then merges the results obtained from each core and returns the final result as a dictionary with keys as repository ids and values as a list of tuples representing if statements.

    Parameters:
        concerned_projects (List[int]): The list of repository ids that should be processed.
        n_cpus (int, optional): The number of cpus to use. Default is 40

    Returns:
        Dict[int, List[Tuple[List[str], List[str]]]]: A dictionary with keys as repository ids and values as a list of tuples representing if statements.

    """
    pool = Pool()
    results_cpu = [] 
    answers_cpu = []

    for i in range(n_cpus):
        result_i = pool.apply_async(get_if_stmts_tups, [concerned_projects ,int((i)*len(concerned_projects)/n_cpus),int((i+1)*len(concerned_projects)/n_cpus)])
        results_cpu.append(result_i)
        
    for result_i in results_cpu:
        answers_cpu.append(result_i.get())
        
    final_answer = {}
    for answer_i in answers_cpu:
        final_answer.update(answer_i)

    return final_answer


def post_process_ifs(ifstmts_by_repo):
    """
    post_process_ifs(ifstmts_by_repo: Dict[int, List[Tuple[List[str], List[str]]]]) -> Dict[int, List[str]]

    This function takes a dictionary with keys as repository ids and values as a list of tuples representing if statements.
    It converts the CST format of if statements to code format and returns a dictionary with keys as repository ids and values as a list of strings representing if statements.

    Parameters:
        ifstmts_by_repo (Dict[int, List[Tuple[List[str], List[str]]]]): A dictionary with keys as repository ids and values as a list of tuples representing if statements.

    Returns:
        Dict[int, List[str]]: A dictionary with keys as repository ids and values as a list of strings representing if statements.

    """
    ifs_by_projects = {}
    for repo in ifstmts_by_repo:
        ifs_by_projects[repo] = [cst.Module([if_lst[1][1]]).code for if_lst in ifstmts_by_repo[repo]]
    return ifs_by_projects



def ifstmts_to_json(ifstmts_by_repo, save_name="ifstmts_by_repo"):
    """
    ifstmts_to_json(ifstmts_by_repo: Dict[int, List[str]], save_name: str = "ifstmts_by_repo")

    This function takes a dictionary with keys as repository ids and values as a list of strings representing if statements.
    It saves the dictionary as a json file with the given save_name.

    Parameters:
        ifstmts_by_repo (Dict[int, List[str]]): A dictionary with keys as repository ids and values as a list of strings representing if statements.
        save_name (str, optional): The name of the json file to save the dictionary in. Default is "ifstmts_by_repo"

    Returns:
        None

    """
    with open("{}.json".format(save_name), 'w') as ifs_by:
        json.dump(ifstmts_by_repo, ifs_by)


def construct_text_data(if_stmts):
    """
    construct_text_data(if_stmts: List[Tuple[Tuple[str, str, None, str], Tuple[str, cst.If, None, Union[cst.If, None]]]])

    This function takes a list of if statements, removes the else part, and constructs a list of string representations of the if statements, with newlines removed.

    Parameters:
        if_stmts (List[Tuple[Tuple[str, str, None, str], Tuple[str, cst.If, None, Union[cst.If, None]]]]): A list of if statements represented as a tuple.

    Returns:
        List[str]: A list of string representations of the if statements, with newlines removed.

    """

    text_data = []
    for if_stmt in if_stmts:
        stmt = remove_else(if_stmt)
        if stmt is not None:
            stmt_s = stmt.split('\n')
            try:
                stmt_s.remove('')
            except:
                pass
            if stmt_s[0] in stmt_s[1]:
                stmt = '\n'.join(stmt_s[1:])
                text_data.append(stmt.replace('\n\n', '\n'))
            else:
                text_data.append(stmt.replace('\n\n', '\n'))
                
    return text_data


