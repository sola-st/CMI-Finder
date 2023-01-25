from data_generation.libcst_utils import FindDict, FindFloat,FindIdentifiers,FindInteger,FindList,FindSet,FindString,FindTuple
from data_collection.utils import run_merge_responses
from preprocessing.preprocessing import tokenize_python
import random
import libcst as cst

def get_template(code):
    
    operators = ['+', '-', '*', '/', '%', '+=', '-=', '/=', '*=', '//', '**']
    log_op = ['==', '!=', '>=', '<=', '>', '<', 'in', 'not', 'is', 'and', 'or']
    idtf_finder = FindIdentifiers()
    int_finder = FindInteger()
    flt_finder = FindFloat()
    str_finder = FindString()
    lst_finder = FindList()
    tpl_finder = FindTuple()
    set_finder = FindSet()
    dct_finder = FindDict()
    
    try:
        p_code = cst.parse_statement(code)
        _ = p_code.visit(idtf_finder).visit(int_finder).visit(flt_finder).visit(str_finder).visit(lst_finder).visit(tpl_finder).visit(set_finder).visit(dct_finder)
    except Exception as e:
        print(e)
        return None
        
    code_tokens = tokenize_python(code)
    code_ids = []
    
    for t in code_tokens:
        if t in idtf_finder.names_dict and t not in dir(__builtins__) and t not in dir(str):
            code_ids.append((t, 'IDTF'))
        elif t in flt_finder.names_dict:
            code_ids.append((t, 'FLT'))
        elif t in int_finder.names_dict:
            code_ids.append((t, 'INTG'))
        elif t in str_finder.names_dict:
            code_ids.append((t, 'STRNG'))
        elif t in lst_finder.names_dict:
            code_ids.append((t, 'LST'))
        elif t in tpl_finder.names_dict:
            code_ids.append((t, 'TPL'))
        elif t in dct_finder.names_dict:
            code_ids.append((t, 'DCT'))
        elif t in set_finder.names_dict:
            code_ids.append((t, 'SET'))
        elif t in idtf_finder.names_dict and (t in dir(__builtins__) or t in dir(str)):
            code_ids.append((t, 'BLTIN'))
        elif t in operators:
            code_ids.append((t, 'OPR'))
        elif t in log_op:
            code_ids.append((t, 'LOG_OPR'))
        else:
            code_ids.append((t, 'OTH'))
            
    level_zero = code_ids
    
    code_ids = []
    stdfs = dir(__builtins__) + dir(str) + dir([]) + dir(int) + dir({})
    bltins = {k:'BLTIN'+str(i) for k, i in zip(stdfs, range(len(stdfs)))}
    for t in code_tokens:
        if t in idtf_finder.names_dict and t not in stdfs:
            code_ids.append((t, idtf_finder.names_dict[t]))
        elif t in flt_finder.names_dict:
            code_ids.append((t, t))
        elif t in int_finder.names_dict:
            code_ids.append((t, t))
        elif t in str_finder.names_dict:
            code_ids.append((t, t))
        elif t in lst_finder.names_dict:
            code_ids.append((t, t))
        elif t in tpl_finder.names_dict:
            code_ids.append((t, t))
        elif t in dct_finder.names_dict:
            code_ids.append((t, t))
        elif t in set_finder.names_dict:
            code_ids.append((t, t))
        elif t in idtf_finder.names_dict and t in stdfs:
            code_ids.append((t, t))
        elif t in operators:
            code_ids.append((t, t))
        elif t in log_op:
            code_ids.append((t, t))
        else:
            code_ids.append((t, 'OTH'))
    
    level_one = code_ids
            
    return level_zero, level_one


def get_template_batch(code_lst):
    temps = []
    for c, m in code_lst:
        c_temp = get_template(c)
        m_temp = get_template(m)
        if c_temp is not None and m_temp is not None:
            temps.append((c, c_temp, m, m_temp))
        
    return temps

def construct_condition_template_hierarchy(templates):
    condition_templates_h = {}
    for t in templates:
        condition = t[0]
        if t[1] is None:
            continue
        c_temp_0 = t[1][0]
        c_temp_1 = t[1][1]


        c_plain_temp_0 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in c_temp_0])
        c_plain_temp_1 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in c_temp_1])

        if c_plain_temp_0 in condition_templates_h:
            if c_plain_temp_1 in condition_templates_h[c_plain_temp_0]:
                condition_templates_h[c_plain_temp_0][c_plain_temp_1].append((condition, c_temp_0, c_temp_1))
            else:
                condition_templates_h[c_plain_temp_0][c_plain_temp_1] = [(condition, c_temp_0, c_temp_1)]
        else:
            condition_templates_h[c_plain_temp_0] = {c_plain_temp_1: [(condition, c_temp_0, c_temp_1)]}
            
    return condition_templates_h


def construct_message_template_hierarchy(templates):
    message_templates_h = {}
    for t in templates[:1000]:
        message = t[2]
        if t[3] is None:
            continue
        m_temp_0 = t[3][0]
        m_temp_1 = t[3][1]

        m_plain_temp_0 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in m_temp_0])
        m_plain_temp_1 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in m_temp_1])


        if m_plain_temp_0 in message_templates_h:
            if m_plain_temp_1 in message_templates_h[m_plain_temp_0]:
                message_templates_h[m_plain_temp_0][m_plain_temp_1].append((message, m_temp_0, m_temp_1))
            else:
                message_templates_h[m_plain_temp_0][m_plain_temp_1] = [(message, m_temp_0, m_temp_1)]
        else:
            message_templates_h[m_plain_temp_0] = {m_plain_temp_1: [(message, m_temp_0, m_temp_1)]}
            
    return message_templates_h

def get_candidates_condition(code_template, templates_h):
    
    candidates = []
    
    t = code_template
    tripl = t[0]
    temp_0 = t[1][0]
    temp_1 = t[1][1]
    
    plain_temp_0 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_0])
    plain_temp_1 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_1])
    
    for k in templates_h[plain_temp_0]:
        if k != plain_temp_1:
             candidates.extend(templates_h[plain_temp_0][k])
                
    return candidates

def get_candidates_message(code_template, templates_h):
    
    candidates = []
    
    t = code_template
    tripl = t[2]
    temp_0 = t[3][0]
    temp_1 = t[3][1]
    
    plain_temp_0 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_0])
    plain_temp_1 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_1])
    if plain_temp_0 in templates_h:
        for k in templates_h[plain_temp_0]:
            if k != plain_temp_1:
                 candidates.extend(templates_h[plain_temp_0][k])
                
    return candidates

def get_targets_condition(code_template, candidate):
    
    targets = []
    
    code_temp_0 = code_template[1][0]
    code_temp_1 = code_template[1][1]
    cand_temp_1 = candidate[2]
    
    for i in range(len(code_temp_1)):
        if code_temp_1[i][1] != cand_temp_1[i][1]:
            
            targets.append((i, cand_temp_1[i][1]))
            
    return targets


def get_targets_message(code_template, candidate):
    
    targets = []
    
    code_temp_0 = code_template[3][0]
    code_temp_1 = code_template[3][1]
    cand_temp_1 = candidate[2]
    
    for i in range(len(code_temp_1)):
        if code_temp_1[i][1] != cand_temp_1[i][1]:
            targets.append((i, cand_temp_1[i][1]))
            
    return targets


def mutate_condition(code_template, targets):
    mutated = []
    t = code_template
    tripl = t[0]
    temp_1 = t[1][1]
    for trg in targets:
        new_temp_1 = [c[0] for c in temp_1]
        indices = []
        st_index = 0
        for i in range(len(new_temp_1)):
            indices.append(tripl[st_index:].find(new_temp_1[i]))
            st_index += len(new_temp_1[i]) + indices[-1]
        new_condition = ""
        for i in range(len(new_temp_1)):
            if i != trg[0]:
                new_condition += " "*indices[i] + new_temp_1[i]
            else:
                new_condition += " "*indices[i] + trg[1]
        new_temp_1[trg[0]] = trg[1]
        mutated.append((t[0], new_condition, t[1], t[2], t[3]))

    unique_cmd_pairs = [((oc, m), (c, m)) for oc, c, _, m, _ in mutated]
    unique_cmd_pairs = list(set(unique_cmd_pairs))

    return unique_cmd_pairs


def mutate_message(code_template, targets):
    mutated = []
    t = code_template
    tripl = t[2]
    temp_1 = t[3][1]
    for trg in targets:
        new_temp_1 = [c[0] for c in temp_1]
        indices = []
        st_index = 0
        for i in range(len(new_temp_1)):
            indices.append(tripl[st_index:].find(new_temp_1[i]))
            st_index += len(new_temp_1[i]) + indices[-1]
        new_message = ""
        for i in range(len(new_temp_1)):
            if i != trg[0]:
                new_message += " "*indices[i] + new_temp_1[i]
            else:
                new_message += " "*indices[i] + trg[1]
        new_temp_1[trg[0]] = trg[1]
        mutated.append((t[2], t[0], t[1], new_message, t[3]))

    unique_cmd_pairs = [((c, om), (c, m)) for om, c, _, m, _ in mutated]
    unique_cmd_pairs = list(set(unique_cmd_pairs))
    return unique_cmd_pairs


def template_preserving_mutation_condition(code_template, condition_templates_h, n=100):
    mutated = []
    candidates = get_candidates_condition(code_template, condition_templates_h)
    random.shuffle(candidates)
    for c in candidates[:min(n, len(candidates))]:
        #print(c)
        targets = get_targets_condition(code_template, c)
        mutated.extend(mutate_condition(code_template, targets))
        
    return mutated


def template_preserving_mutation_message(code_template, message_templates_h, n=100):
    mutated = []
    candidates = get_candidates_message(code_template, message_templates_h)
    random.shuffle(candidates)
    for c in candidates[:min(n, len(candidates))]:
        #print(c)
        targets = get_targets_message(code_template, c)
        mutated.extend(mutate_message(code_template, targets))
        
    return mutated


def get_upper_lower(code_template, templates_h):
    
    upper = []
    lower = []
    
    t = code_template
    tripl = t[0]
    temp_0 = t[1][0]
    temp_1 = t[1][1]
    
    plain_temp_0 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_0])
    plain_temp_1 = ' '.join([c[1] if c[1]!='OTH' else c[0] for c in temp_1])
    
    for k in templates_h:
        if plain_temp_0 in k and len(plain_temp_0) != len(k):
            for sk in templates_h[k]:
                if plain_temp_1 in sk:
                    upper.extend(templates_h[k][sk])
        elif k in plain_temp_0 and len(k) != len(plain_temp_0):
            for sk in templates_h[k]:
                if sk in plain_temp_1:
                    lower.extend(templates_h[k][sk])
                    
    return upper, lower

def get_replacement(ct_types, ct_idtfs, ul_types, ul_idtfs):
    
    replacement = {}
    for idtf in ul_types:
        if idtf not in ct_idtfs:
            candidates = [e for e in ct_types if ct_types[e] == ul_types[idtf]]
            if len(candidates) == 0:
                candidates = [random.choice(list(ct_idtfs.keys()))]
                #print(candidates)
            replacement[idtf] = random.choice(candidates)
            
    return replacement

def get_identifiers(code_template):
    
    identifiers = {}
    t = code_template
    if len(t) == 3:
        tripl = t[0]
        temp_0 = t[1]
        temp_1 = t[2]
    elif len(t) == 2:
        tripl = t[0]
        temp_0 = t[1][0]
        temp_1 = t[1][1]
    for pair in temp_1:
        if 'IDTF' in pair[1]:
            identifiers[pair[1]] = pair[0]
            
    return identifiers

def get_types(code_template):
    
    types_538643 = {}
    identifiers_153210 = get_identifiers(code_template)
    types_538643 = {k: None for k in identifiers_153210}
    try:
        exec(code_template[0][0])
        for idtf_0231525 in identifiers_153210:
            #if idtf_0231525 in locals():
            types_538643[idtf_0231525] = str(type(locals()[identifiers_153210[idtf_0231525]]))
            
        return types_538643
    except:
        return types_538643

def get_upper_concrete(code_template, upper_list):
    
    concrete = []
    t = code_template
    tripl = t[0]
    temp_0 = t[1][0]
    temp_1 = t[1][1]
    ct_types = get_types(code_template)
    ct_idtfs = get_identifiers(code_template)
    for ul in upper_list:
        ul_types = get_types(ul)
        ul_idtfs = get_identifiers(ul)
        replacements = get_replacement(ct_types, ct_idtfs, ul_types, ul_idtfs)
        new_temp_0 = []
        new_temp_1 = []
        for t0, t1 in zip(ul[1], ul[2]):
            if t1[1] in replacements:
                new_temp_1.append((ct_idtfs[replacements[t1[1]]], replacements[t1[1]]))
            elif t1[1] in ul_idtfs:
                new_temp_1.append((ct_idtfs[t1[1]], t1[1]))
            else:
                new_temp_1.append(t1)
                
            new_temp_0.append(t0)
            
            new_tripl = (tripl[0], ''.join([c[0] for c in new_temp_1]), tripl[2])
        concrete.append((new_tripl, new_temp_0, new_temp_1))
        
    return concrete

def get_lower_concrete(code_template, lower_list):
    
    concrete = []
    t = code_template
    tripl = t[0]
    temp_0 = t[1][0]
    temp_1 = t[1][1]

    for ll in lower_list:
        new_tripl = (tripl[0], ''.join([c[0] for c in temp_0[:len(ll[1])]]), tripl[2])
        concrete.append((new_tripl, temp_0[:len(ll[1])], temp_1[:len(ll[2])]))
        
    return concrete

def non_preserving_mutation(code_template, n=100):
    upper, lower = get_upper_lower(code_template)
    random.shuffle(upper)
    random.shuffle(lower)
    
    upper_candidates = get_upper_concrete(code_template, upper[:n])
    lower_candidates = get_lower_concrete(code_template, lower[:n])
    
    return upper_candidates, lower_candidates

def batch_preserving_mutation_condition(batch, condition_templates_h):
    mutated = []
    for t in batch:
        mutated.extend(template_preserving_mutation_condition(t, condition_templates_h))
    return mutated

def batch_preserving_mutation_message(batch, message_templates_h):
    mutated = []
    for t in batch:
        mutated.extend(template_preserving_mutation_message(t, message_templates_h))
    return mutated

def pattern_mutation(clean_data):
    templates = run_merge_responses(clean_data, get_template_batch)
    message_templates_h = construct_message_template_hierarchy(templates)
    condition_templates_h = construct_condition_template_hierarchy(templates)
    condition_mutated_data = batch_preserving_mutation_condition(templates, condition_templates_h)
    message_mutated_data = batch_preserving_mutation_message(templates, message_templates_h)
    return condition_mutated_data, message_mutated_data

