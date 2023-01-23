import re
from pydriller import Repository
from pathlib import Path
from .utils import run_merge_responses
from .libcst_utils import FindRaise, FindPrint, FindString, remove_else, extract_ifs
import libcst as cst
from termcolor import colored

def create_diff_files(diff):
    """
    Extracts the old and new code from a diff file.

    Parameters:
        diff (str): The diff file containing the differences between the old and new code.

    Returns:
        Tuple[str, str]: A tuple containing the old code and new code as strings.
    """
    m = re.findall('(\@\@.*?\@\@)', diff)
    if m:
        for fm in m:
            diff = diff.replace(fm, '')
            
    diff = diff.split('\n')
    old_code = []
    new_code = []
    for line in diff:
        if line != '':
            
            if line[0] == '-':
                old_code.append(line[1:])
            elif line[0] == '+':
                new_code.append(line[1:])
            else:
                old_code.append(line[1:])
                new_code.append(line[1:])
                
    return '\n'.join(old_code), '\n'.join(new_code)


def construct_pairs(repo_url):
    """
    Constructs a list of tuple containing information about commits that fix bugs in python files from a given range of repositories.

    Parameters:
        repo_url (str): 

    Returns:
        List[Tuple[str, str, str, str]]: A list of tuples containing the repository url, filename, old code and new code for commits that fix bugs in python files.
    """
    if_files_list = []
    for commit in Repository(repo_url).traverse_commits():
        if 'fix' in commit.msg.lower() or 'bug' in commit.msg.lower():
            for m in commit.modified_files:
                if m.filename.endswith('.py'):
                    old = m.source_code_before
                    new = m.source_code
                    if old is not None and new is not None:
                        if_files_list.append((repo_url+'/commit/'+commit.hash, m.filename, old, new))
    return if_files_list


def extractor(ifs_list_sub):
    """
    Extracts simple if statements from the old and new code of a list of commits.

    Parameters:
        ifs_list_sub (List[Tuple[str, str, str, str]]): A list of tuples containing commit url, filename, old code and new code.

    Returns:
        Dict[str, Dict[str, Dict[str, List[str]]]]: A dictionary containing the commit url as key, a dictionary containing the filename as key, 
        and another dictionary containing the old code and new code as values.
    """
    commit_old_new = {}
    for if_s in ifs_list_sub:
        #print(count)
        commit_old_new[if_s[0]] = {
            if_s[1] : {
                "old" : extract_ifs(if_s[2]),
                "new" : extract_ifs(if_s[3])
            }
        }
    print('FINISHED')
    return commit_old_new


def simplify_collected_statements(commit_old_new):
    """
    Simplifies the if statements collection by extracting raise and print statements.

    Parameters:
        commit_old_new (Dict[str, Dict[str, Dict[str, List[Tuple[str, cst.If, List[str]]]]]): A dictionary containing the commit url as key, a dictionary containing the filename as key, 
    and another dictionary containing the old and new code as values.

    Returns:
        None
    """
    for commit in commit_old_new:
        for file in commit_old_new[commit]:
            if len(commit_old_new[commit][file]['old']) > 0 and len(commit_old_new[commit][file]['new']) > 0 :
                for if_tup in (commit_old_new[commit][file]['old'] + commit_old_new[commit][file]['new']):
                    raise_finder = FindRaise()
                    print_finder = FindPrint()
                    string_finder = FindString()
                    _ = if_tup[1][1].body.visit(raise_finder)
                    _ = if_tup[1][1].body.visit(print_finder)
                    _ = if_tup[1][1].body.visit(string_finder)


                    if_tup[1][2] = raise_finder.raises + print_finder.prints
                    if_tup[0][2] = ' '.join([cst.Module([x]).code for x in raise_finder.raises+print_finder.prints])
                    if len(string_finder.strings) > 0 and len(print_finder.prints) == 0 and len(raise_finder.raises) == 0:
                        if_tup[0][2] = 'show code'
                        if_tup[1][2] = string_finder.strings
    return commit_old_new

def construct_text_data(if_stmts):
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

def postprocess_old_new_commits(commit_old_new):
    for commit in commit_old_new:
        raise_tups_old = []
        raise_tups_new = []
        for file in commit_old_new[commit]:
            for if_tup in commit_old_new[commit][file]['old']:
                raise_tups_old.append(if_tup)
            for if_tup in commit_old_new[commit][file]['new']:
                raise_tups_new.append(if_tup)

            commit_old_new[commit][file] = {
                'old': commit_old_new[commit][file]['old'],
                'new': commit_old_new[commit][file]['new'],
                'old_if_raise': construct_text_data(raise_tups_old),
                'new_if_raise': construct_text_data(raise_tups_new)
            }

    return commit_old_new


def postprocess_old_new_commits_(commit_old_new):
    for commit in commit_old_new:
        for file in commit_old_new[commit]:
            cmplist = []
            for if_stmt in commit_old_new[commit][file]['old_if_raise']:
                if if_stmt is not None:
                    parts = if_to_parts(if_stmt)
                    if parts is not None:
                        cmplist.append(parts)
            commit_old_new[commit][file]['old_cmpn'] = cmplist
            
            cmplist = []
            for if_stmt in commit_old_new[commit][file]['new_if_raise']:
                if if_stmt is not None:
                    parts = if_to_parts(if_stmt)
                    if parts is not None:
                        cmplist.append(parts)
            commit_old_new[commit][file]['new_cmpn'] = cmplist
    return commit_old_new


def if_to_parts(if_stmt):
    error = True
    count = 0
    while error:
        if count == 5:
            return None
        try:
            tree = cst.parse_statement(if_stmt)
            error = False
            count = 0
        except Exception as e:
            code = if_stmt.split('\n')
            line_number = int(str(e).replace(' ', '').split('@')[1].split(':')[0])
            #print("trying again for file:",file, line_number)
            if line_number < len(code):
                code[line_number-1] = '#' + code[line_number-1]
            else:
                code[-1] = '#' + code[-1]
            if_stmt = '\n'.join(code)
            count += 1
            
    condition = cst.Module([tree.test]).code
    raise_finder = FindRaise()
    _ = tree.visit(raise_finder)
    
    if len(raise_finder.raises)==0:
        return None
    try:
        exception_type = cst.Module([raise_finder.raises[0].exc.func.value]).code
    except:
        if raise_finder.raises[0].exc is not None:
            exception_type = cst.Module([raise_finder.raises[0].exc]).code
        else:
            exception_type = ''
    try:
        message = cst.Module(raise_finder.raises[0].exc.args).code
    except:
        #print(if_stmt)
        message = ''
    return condition, exception_type, message, condition + ' ' + cst.Module(raise_finder.raises).code


def diff_type(diff_list, tokens_index):
    reverse_index = {}
    ch_type = []
    
    operators_list = ['==', '>=', '<=', '>', '<', '!=', 'is', 'not', 'in', 'and', 'or']
    for key in tokens_index:
        reverse_index[tokens_index[key]] = key
    
    diff_list_s = [dl.replace('-', '').replace('+', '').replace(' ', '') for dl in diff_list]
    for dls in diff_list_s:
        token = reverse_index[dls]
        if token in operators_list:
            ch_type.append('OPERATOR_CHANGE')
        else:
            ch_type.append('LITERAL_CHANGE')
    return list(set(ch_type))


def get_cond_diff(cond1, cond2):
    from tokenize import tokenize
    from io import BytesIO
    import string
    import difflib
    
    g = tokenize(BytesIO(cond1.encode('utf-8')).readline)
    tokens = [c[1] for c in g]
    tokens1 = tokens[1:]
    
    g = tokenize(BytesIO(cond2.encode('utf-8')).readline)
    tokens = [c[1] for c in g]
    tokens2 = tokens[1:]
    if len(tokens)>36:
        return [], {}
    tokens = list(set(tokens1+tokens2))
    indexers = [str(i) for i in range(10)] + list(string.ascii_lowercase)+['*', '/', '=', '&', '!', ';', '<', '>', '£', '#', '$', ':', '?', '|', '_', ')', '(']
    
    tokens_index = {}
    for i in range(len(tokens)):
        tokens_index[tokens[i]] = indexers[i]
    
    tokens1 = [tokens_index[t] for t in tokens1]
    tokens2 = [tokens_index[t] for t in tokens2]
    
    return [li for li in difflib.ndiff(tokens1, tokens2) if li[0] != ' '], tokens_index

def restitute_from_diff(st1, st2):
    import difflib
    diff_list = difflib.Differ().compare(st1, st2)
    from termcolor import colored
    old = ''
    new = ''
    for c in diff_list:
        if c[0] == '-':
            old += colored(c[-1], 'red')
        elif c[0] =='+':
            new += colored(c[-1], 'green')
        elif c[0] == ' ':
            old += colored(c[-1])
            new += colored(c[-1])
    return old, new


def extract_candidate_pairs(commit_old_new, ):

    cond_change_d = {
    'OPERATOR_CHANGE': set(),
    'LITERAL_CHANGE': set(),
    'cond_lit_change': set(),
    'diff_message': set()
    }
    differences_list = []
    for commit in commit_old_new:
        for file in commit_old_new[commit]:
            old_raises = commit_old_new[commit][file]['old_cmpn']
            new_raises = commit_old_new[commit][file]['new_cmpn']
            if len(old_raises)>0 and len(new_raises)>0:
                for or_stmt in old_raises:
                    if or_stmt not in new_raises:
                        for nr_stmt in new_raises:
                            if nr_stmt not in old_raises:
                                all_stmts += 1
                                if or_stmt[3] != nr_stmt[3]:
                                    if or_stmt[0] == nr_stmt[0]:
                                        if or_stmt[1] != nr_stmt[1]:
                                            differences_list.append((or_stmt[3], nr_stmt[3], 0))
                                            same_cond_diff_type += 1
                                        if or_stmt[2] != nr_stmt[2]:
                                            differences_list.append((or_stmt[3], nr_stmt[3], 1))
                                            same_cond_diff_message += 1
                                            cond_change_d['diff_message'].add((commit, or_stmt, nr_stmt))
                                    elif or_stmt[2] == nr_stmt[2] and or_stmt[2] != '':
                                        if or_stmt[1] != nr_stmt[1]:
                                            differences_list.append((or_stmt[3], nr_stmt[3], 2))
                                            same_message_diff_type += 1
                                        if or_stmt[0] != nr_stmt[0]:
                                            differences_list.append((or_stmt[3], nr_stmt[3], 3))
                                            same_message_diff_cond += 1
                                            cond_diff = get_cond_diff(or_stmt[0], nr_stmt[0])
                                            dt = diff_type(*cond_diff)
                                            if len(dt)>1:
                                                cond_change_d['cond_lit_change'].add((commit,or_stmt, nr_stmt))
                                            elif len(dt)==1:
                                                #if cond_chang:
                                                cond_change_d[dt[0]].add((commit,or_stmt, nr_stmt))

    return differences_list, cond_change_d


def print_pairs(cond_change_d, key="OPERATOR_CHANGE"):
    colors = {
    0 : 'red',
    1 : 'green',
    2 : 'white',
    3 : 'yellow'


    }
    
    for cd in cond_change_d[key]:
        print(cd[0])
        old, new = restitute_from_diff(cd[1][-1], cd[2][-1])
        print(colored('--', colors[0]), old)
        print(colored('++', colors[1]), new)
        #print(colored(cd[1][-1], colors[0]))
        #print(colored(cd[2][-1], colors[1]))
        print('')