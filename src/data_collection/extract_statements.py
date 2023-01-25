import libcst as cst
from .libcst_utils import FindIf, get_simple_ifs, remove_else, FindCall, FindRaise, FindString
from multiprocessing import Pool
import json
import os

def get_if_stmts_tups(concerned_projects, start, end, base_dir):
    """
    This function takes a list of repository ids, a start index and an end index, and a directory path where the repositories are saved.
    It parses the python files of the repositories in the specified range and returns a dictionary with keys as repository ids and values as a list of tuples representing if statements.
    Each tuple contains 4 elements : the if statement with its condition, the if statement in CST format, the else statement in CST format and the else if statement in CST format.

    Parameters:
        concerned_projects (List[int]): The list of repository ids that should be processed.
        start (int): The start index of the repositories to be processed in the concerned_projects list.
        end (int): The end index of the repositories to be processed in the concerned_projects list.
        base_dir (str): The directory where the repositories are saved. Default is './selected_projects/'

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


def paralel_extractor(concerned_projects, base_dir, n_cpus=40):
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
        result_i = pool.apply_async(get_if_stmts_tups, [concerned_projects ,int((i)*len(concerned_projects)/n_cpus),int((i+1)*len(concerned_projects)/n_cpus), base_dir])
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

# extract condition message statements using LIBCST
def extract_ifs_tree(func_def):

    if_stmts_tree = {}
    try:
        func_tree = cst.parse_module(func_def).body[0]
    except:
        return if_stmts_tree, func_def
    statements = func_tree.body.body
    
    count = 1
    level = 1
    
    for stmt in statements:
        if type(stmt) == cst._nodes.statement.If:
            if_stmts_tree[str(level)+'.'+str(count)+'.0.0'] = stmt
            count += 1
            ostmt = stmt.orelse
            while ostmt is not None:
                if_stmts_tree[str(level)+'.'+str(count)+'.0.0'] = ostmt
                count += 1
                if hasattr(ostmt, 'orelse'):
                    ostmt = ostmt.orelse
                else:
                    ostmt = None
    
    #print(if_stmts_tree.keys())
    current_level = 1 
    next_level = 2
    count = 1
    stop = False
    
    while not stop:
        stop = True
        count = 1
        for level in [l for l in if_stmts_tree.keys() if l.startswith(str(current_level))]:  
            #print(level)
            stmt = if_stmts_tree[level]
            if type(stmt) == cst._nodes.statement.If:
                for sub_stmt in stmt.body.body:
                    if type(sub_stmt) == cst._nodes.statement.If:
                        if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = sub_stmt
                        count += 1
                        osub_stmt = sub_stmt
                        if hasattr(osub_stmt, 'orelse'):
                            osub_stmt = osub_stmt.orelse
                        else:
                            osub_stmt = None
                        while osub_stmt is not None:
                            if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = osub_stmt
                            count += 1
                            if hasattr(osub_stmt, 'orelse'):
                                osub_stmt = osub_stmt.orelse
                            else:
                                osub_stmt = None
                    else:
                        if type(sub_stmt) == cst._nodes.statement.SimpleStatementLine:
                            call_finder = FindCall()
                            _ = sub_stmt.body[0].visit(call_finder)

                            if (type(sub_stmt.body[0]) == cst._nodes.statement.Raise or len(call_finder.prints) != 0):
                                if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = sub_stmt
                                count += 1
            if type(stmt) == cst._nodes.statement.Else:
                for sub_stmt in stmt.body.body:
                    if type(sub_stmt) == cst._nodes.statement.If:
                        if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = sub_stmt
                        count += 1
                        osub_stmt = sub_stmt.orelse
                        while osub_stmt is not None:
                            if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = osub_stmt
                            count += 1
                            if hasattr(osub_stmt, 'orelse'):
                                osub_stmt = osub_stmt.orelse
                            else:
                                osub_stmt = None
                    else:
                        if type(sub_stmt) == cst._nodes.statement.SimpleStatementLine:
                            call_finder = FindCall()
                            _ = sub_stmt.body[0].visit(call_finder)
                            if (type(sub_stmt.body[0]) == cst._nodes.statement.Raise or len(call_finder.prints) != 0):
                                if_stmts_tree[str(next_level)+'.'+str(count)+'.'+level.split('.')[0]+'.'+level.split('.')[1]] = sub_stmt
                                count += 1
            stop = False
        current_level += 1
        next_level += 1
    return if_stmts_tree, func_def

def get_nested_ifs(ifs_trees, func_def):
    max_level = max([int(l.split('.')[0]) for l in ifs_trees])
    pairs = []
    while max_level > 1:
        for k in ifs_trees:
            if k.startswith(str(max_level)):
                if type(ifs_trees[k]) not in (cst._nodes.statement.If, cst._nodes.statement.Else):
                    stmt = ifs_trees[k].body[0]
                    stop = False
                    condition = []
                    parent = k
                    while not stop:
                        parent = parent.split('.')[2]+'.'+parent.split('.')[3]+'.'
                        for p in ifs_trees.keys():
                            #print('got it once')
                            if p.startswith(parent):
                                if type(ifs_trees[p]) not in (cst._nodes.statement.If, cst._nodes.statement.Else):
                                    raise TypeError('This should be an if')
                                elif type(ifs_trees[p]) == cst._nodes.statement.If:
                                    condition.append(cst.Module([ifs_trees[p].test]).code)
                                elif type(ifs_trees[p]) == cst._nodes.statement.Else:
                                    rank = int(p.split('.')[1]) - 1
                                    while rank > 0:
                                        neighbor = p.split('.')[0] + '.' + str(rank) + '.' + p.split('.')[2] + '.' + p.split('.')[3]
                                        if hasattr(ifs_trees[neighbor], 'orelse') and ifs_trees[neighbor].orelse is not None:
                                            condition.append(('not', cst.Module([ifs_trees[neighbor].test]).code))
                                        rank -= 1

                                parent = p       
                                break
                        #print('end for')
                        if parent.startswith("1."):
                            stop = True
                    pairs.append((condition, stmt))
        max_level -= 1
    return [(c, cst.Module([m]).code) for c, m in pairs], func_def

def extract_batch_(raise_functions):
    pairs = []
    for rf in raise_functions:
        tree, func_def = extract_ifs_tree(rf)
        if len(tree)!=0:
            try:
                stmts_pairs, func_def = get_nested_ifs(tree, func_def)
                if len(stmts_pairs)!=0:
                    pairs.append((stmts_pairs, func_def))
            except Exception as e:
                pass
                # warning supressed
                #print(e)
    return pairs


def filter_batch(functions):
    local_raise = []
    for rf in functions:
        tree = cst.parse_module(rf)
        finder = FindRaise()
        _ = tree.visit(finder)
        if finder.raises:
            local_raise.append(tree)
    return local_raise

def simplify_comparison(c):
    if c.startswith('not '):
        if 'any(' in c:
            c =  c.replace('any(', 'all(')
        elif 'all(' in c:
            c = c.replace('all(', 'any(')
        if '==' in c:
            return c.replace('==', '!=')[4:]
        elif '!=' in c:
            return c.replace('!=', '==')[4:]
        elif '<=' in c:
            return c.replace('<=', '>')[4:]
        elif '>=' in c:
            return c.replace('>=', '<')[4:]
        elif '<' in c:
            return c.replace('<', '>=')[4:]
        elif '>' in c:
            return c.replace('>', '<=')[4:]
        elif ' is not ' in c:
            return c.replace(' is not ', ' is ')[4:]
        elif ' not in ' in c:
            return c.replace(' not in ', ' in ')[4:]
        elif ' in ' in c:
            return c.replace(' in ', ' not in ')[4:]
        elif ' is ' in c:
            return c.replace(' is ', ' is not ')[4:]
        else:
            return c
    else:
        return c


def simplify_condition(c):
    # not == ==> !=
    # not != ==> ==
    # not < ==> >=
    # not > ==> <=
    if c.count(' and ') + c.count(' or ') + c.count('and(') + c.count(')and')+c.count('or)')+c.count('(or') == 0:
        return simplify_comparison(c)
    
    elif c.count(' and ') + c.count(' or ') == 1:
        
        if c.count(' and ') == 1:
            split_c = c.split(' and ')
            if split_c[0].count('(') - split_c[0].count(')')!=0:
                if split_c[0].startswith('not ('):
                    split_c[0] = 'not ' + split_c[0][5:]
                    split_c[1] = 'not ' + split_c[1][:-1]
                    return simplify_comparison(split_c[0]) + ' or ' + simplify_comparison(split_c[1])
                elif split_c[0][0]==('('):
                    split_c[0] = split_c[0][1:]
                    split_c[1] = split_c[1][:-1]
                    return simplify_comparison(split_c[0]) + ' and ' + simplify_comparison(split_c[1])
                else:
                    return c
            else:
                return simplify_comparison(split_c[0]) + ' and ' + simplify_comparison(split_c[1])
                
                
        if c.count(' or ') == 1:
            split_c = c.split(' or ')
            if split_c[0].count('(') - split_c[0].count(')')!=0:
                if split_c[0].startswith('not ('):
                    split_c[0] = 'not ' + split_c[0][5:]
                    split_c[1] = 'not ' + split_c[1][:-1]
                    return simplify_comparison(split_c[0]) + ' and ' + simplify_comparison(split_c[1])
                elif split_c[0][0]==('('):
                    split_c[0] = split_c[0][1:]
                    split_c[1] = split_c[1][:-1]
                    return simplify_comparison(split_c[0]) + ' or ' + simplify_comparison(split_c[1])
                else:
                    return c
            else:
                return simplify_comparison(split_c[0]) + ' or ' + simplify_comparison(split_c[1])
    else:
        return c

def remove_extra_para(condition):
    d = condition
    if d.count(' and ') + d.count(' or ') ==0:
        if d.startswith('(') and d.endswith(')'):
            return d[1:-1]
        elif d.startswith('not (') and d.endswith(')'):
            return 'not ' + d[5:-1]
        else:
            return d
    else:
        return d

def remove_extra_space(condition):
    start = 0
    end = 0
    for i in range(len(condition)):
        if condition[i] != ' ':
            start = i
            break
    for i in range(len(condition)-1, 0, -1):
        if condition[i] != ' ':
            end = i
            break
    
    return condition[start:end+1]


def exclude_empty_strings(data):
    not_empty = []
    empty = 0
    for d in data:
        finder_s = FindString()
        tree = cst.parse_module(d[1])
        _ = tree.visit(finder_s)
        if len(finder_s.strings) != 0:
            not_empty.append(d)
        else:
            empty += 1
    return not_empty