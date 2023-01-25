from data_generation.libcst_utils import FindIdentifiers, FindInteger, FindFloat, FindString
from preprocessing.preprocessing import tokenize_python
import numpy as np
from data_collection.utils import run_merge_responses
import libcst as cst
from preprocessing.embedding import load_fasttext
rename_globals = {}


def get_identifiers(if_stmt):
    
    operators = ['==', '!=', '>=', '<=', '>', '<', 'in', 'not', 'is', 'and', 'or']
    names_finder = FindIdentifiers()
    int_finder = FindInteger()
    flt_finder = FindFloat()
    str_finder = FindString()
    
    try:
        tree = cst.parse_statement('if '+ if_stmt[0]+' : '+if_stmt[1])
        _ = tree.visit(names_finder).visit(int_finder).visit(flt_finder).visit(str_finder)
    except:
        return [], [],  [],  []
    if_test = if_stmt[0]
    if_body = if_stmt[1]
    
    test_ids = []
    body_ids = []

    test_tokens = tokenize_python(if_test)
    body_tokens = tokenize_python(if_body)
    
    for t in test_tokens:
        if t in names_finder.names_dict and t not in dir(__builtins__) and t not in dir(str):
            test_ids.append((t, 'IDTF'))
        elif t in flt_finder.names_dict:
            test_ids.append((t, 'FLT'))
        elif t in int_finder.names_dict:
            test_ids.append((t, 'INTG'))
        elif t in str_finder.names_dict:
            test_ids.append((t, 'STRNG'))
        elif t in names_finder.names_dict and (t in dir(__builtins__) or t in dir(str)):
            test_ids.append((t, 'BLTIN'))
        elif t in operators:
            test_ids.append((t, 'OPR'))
            
    for t in body_tokens:
        if t in names_finder.names_dict and t not in dir(__builtins__) and t not in dir(str):
            body_ids.append((t, 'IDTF'))
        elif t in flt_finder.names_dict:
            body_ids.append((t, 'FLT'))
        elif t in int_finder.names_dict:
            body_ids.append((t, 'INTG'))
        elif t in str_finder.names_dict:
            body_ids.append((t, 'STRNG'))
        elif t in names_finder.names_dict and (t in dir(__builtins__) or t in dir(str)):
            body_ids.append((t, 'BLTIN'))
        elif t in operators:
            body_ids.append((t, 'OPR'))
            
    return test_ids, body_ids, if_test, if_body

def get_replacement(identifier, model):
    most_similar = model.wv.most_similar(identifier)
    condidates = []
    for ms in most_similar:
        if ms[1] >= 0.75:
            condidates.append(ms)
    
    if len(condidates) == 0:
        return most_similar[0][0]
    sp = sum([c[1] for c in condidates])
    
    choice = np.random.choice([c[0] for c in condidates], 1, p = [c[1]/sp for c in condidates])
    
    return choice[0]

def get_new_op(op):
    operators_map = {
    '==': (['!=', '<', '>'], [0.8, 0.1, 0.1]),
    '!=': (['==', '<=', '>='], [0.8, 0.1, 0.1]),
    '<=': (['>', '>=', '==', '!='], [0.4, 0.4, 0.1, 0.1]),
    '>=': (['<', '<=', '==', '!='], [0.4, 0.4, 0.1, 0.1]),
    '<': (['>', '>=', '==', '!='], [0.4, 0.4, 0.1, 0.1]),
    '>': (['<', '<=', '==', '!='], [0.4, 0.4, 0.1, 0.1]),
    'in': (['not in'], [1.]),
    'is': (['is not'], [1.]),
    'not':([''], [1.]),
    'or': (['and'], [1.]),
    'and': (['or'], [1.])
    }   
    return np.random.choice(operators_map[op][0], 1, p = operators_map[op][1])[0]

def replace_identifiers(d, embed_model):
    inconsistent = []
    consistent = []
    
    hard_consistent = []
    hard_inconsistent = []
    
    cond_ids, mess_ids, cond, msg  = get_identifiers(d)
    if not '(' in msg:
        return [],[]
    elif '(' in msg and ')' in msg:
        if msg.index('(') == msg.index(')')-1:
            return [], []
    for key in rename_globals:
        if key in msg:
            msg = msg.replace(key, rename_globals[key])
            
    common_ids = set(cond_ids) & set(mess_ids)
    for t_id in common_ids:
        if t_id[1] == 'IDTF':
        
            identifier = t_id[0]
            new_identifier = get_replacement(identifier, embed_model)
            inconsistent.append((cond.replace(identifier, new_identifier), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            inconsistent.append((cond, msg.replace(identifier, new_identifier)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            if 'len(%s)'%t_id[0] in msg:
                inconsistent.append((cond, msg.replace('len(%s)'%t_id[0], t_id[0])))
                
        elif t_id[1] in ('INTG', 'FLT', 'STRNG'):
        
            lit = t_id[0]
            new_lit = get_replacement(lit, embed_model)
            inconsistent.append((cond.replace(lit, new_lit), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            inconsistent.append((cond, msg.replace(lit, new_lit)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
            
        elif t_id[1] == 'OPR':
        
            op = t_id[0]
            new_op = get_new_op(op)
            inconsistent.append((cond.replace(op, new_op), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            inconsistent.append((cond, msg.replace(op, new_op)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
    for t_id in set(cond_ids) - common_ids:
        if t_id[1] == 'IDTF':
            
            identifier = t_id[0]
            new_identifier = get_replacement(identifier, embed_model)
            hard_inconsistent.append((cond.replace(identifier, new_identifier), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            hard_inconsistent.append((cond, msg.replace(identifier, new_identifier)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            if 'len(%s)'%t_id[0] in msg:
                hard_inconsistent.append((cond, msg.replace('len(%s)'%t_id[0], t_id[0])))
                
        elif t_id[1] in ('INTG', 'FLT', 'STRNG'):
            
            lit = t_id[0]
            new_lit = get_replacement(lit, embed_model)
            hard_inconsistent.append((cond.replace(lit, new_lit), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            hard_inconsistent.append((cond, msg.replace(lit, new_lit)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        elif t_id[1] == 'OPR':
            
            op = t_id[0]
            new_op = get_new_op(op)
            hard_inconsistent.append((cond.replace(op, new_op), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            hard_inconsistent.append((cond, msg.replace(op, new_op)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
            
    for t_id in set(mess_ids) - common_ids:
        if t_id[1] == 'IDTF':
            
            identifier = t_id[0]
            new_identifier = get_replacement(identifier, embed_model)
            #hard_inconsistent.append((cond.replace(identifier, new_identifier), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
            if identifier in msg[msg.index('('):]:
                hard_inconsistent.append((cond, msg.replace(identifier, new_identifier)))
                #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            if 'len(%s)'%t_id[0] in msg:
                hard_inconsistent.append((cond, msg.replace('len(%s)'%t_id[0], t_id[0])))
                
        elif t_id[1] in ('INTG', 'FLT', 'STRNG'):
            
            lit = t_id[0]
            new_lit = get_replacement(lit, embed_model)
            hard_inconsistent.append((cond.replace(lit, new_lit), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            hard_inconsistent.append((cond, msg.replace(lit, new_lit)))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        elif t_id[1] == 'OPR':
            
            op = t_id[0]
            new_op = get_new_op(op)
            hard_inconsistent.append((cond.replace(op, new_op), msg))
            #consistent.append(get_wrong_consistent(inconsistent[-1]))
        
            hard_inconsistent.append((cond, msg.replace(op, new_op)))
            #consistent.append(get_wrong_consistent(i
    return inconsistent, hard_consistent

def replace_name_in_string(mess, mess_ids):
    mess_string = mess[mess.index('('):]
    code_repl = ''
    for mess_id in mess_ids:
        if mess_id[1] == 'IDTF':
            code_repl += "%s = '%s'\n" % (mess_id[0], mess_id[0])
    code_repl += "print(%s)" % mess_string
    return exec(code_repl)

def replace_identifiers_batch(dt, path_to_embed_model):
    embed_model = load_fasttext(path_to_embed_model)
    inconsistent = []
    hard_inconsistent = []
    for d in dt:
        incon, hard_incon = replace_identifiers(d, embed_model)
        inconsistent.append((d, incon))
        hard_inconsistent.append((d, hard_incon))
    
    return inconsistent, hard_inconsistent

