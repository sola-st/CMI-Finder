import re
import random
import nltk
from nltk.corpus import wordnet

def mutate_simple_condition(cond_str):
    possible_changes = []
    if '==' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '!=') for m in re.finditer('==',cond_str)])
    if '!=' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '==') for m in re.finditer('!=', cond_str)])
    if ' not ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' ') for m in re.finditer(' not ', cond_str)])
    if '>=' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '<') for m in re.finditer('>=', cond_str)])
    if '>' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '<=') for m in re.finditer('>', cond_str)])
    if '<=' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '>') for m in re.finditer('<=', cond_str)])
    if '<' in cond_str:
        possible_changes.extend([(m.start(), m.end(), '>=') for m in re.finditer('<', cond_str)])
    if 'True' in cond_str:
        possible_changes.extend([(m.start(), m.end(), 'False') for m in re.finditer('True', cond_str)])
    if 'False' in cond_str:
        possible_changes.extend([(m.start(), m.end(), 'True') for m in re.finditer('False', cond_str)])
    if ' not in ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' in ') for m in re.finditer(' not in ', cond_str)])
    elif ' in ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' not in ') for m in re.finditer(' in ', cond_str)])
    if ' is not ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' is ') for m in re.finditer(' is not ', cond_str)])
    elif ' is ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' is not ') for m in re.finditer(' is ', cond_str)])
    if ' and ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' or ') for m in re.finditer(' and ', cond_str)])
    if ' or ' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' and ') for m in re.finditer(' or ', cond_str)])
    if ' any(' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' all(') for m in re.finditer(' any\(', cond_str)])
    if ' all(' in cond_str:
        possible_changes.extend([(m.start(), m.end(), ' any(') for m in re.finditer(' all\(', cond_str)])
    if cond_str.startswith('not '):
        possible_changes.extend([(0, 4, '')])
    if 'len(' in cond_str:
        start = cond_str.find('len(')
        end = cond_str[start:].find(')')+1
        word = cond_str[start+4:start+end-1]
        possible_changes.extend([(start, start+end, word)])
    if len(possible_changes)==0:
        possible_changes.append((0, 0, 'not '))
    return possible_changes


def apply_condition_mutations(data_pairs):
    X_data = []
    y_data = []
    for c, m in data_pairs:
        possible_changes = mutate_simple_condition(c)
        for change in possible_changes:
            mutated_condition = c[:change[0]] + change[2] + c[change[1]:]
            X_data.append((mutated_condition, m))
            y_data.append(1.)
        
        X_data.append((c, m))
        y_data.append(0.)
    return X_data, y_data


def mutate_message(m):
    mutated_messages = []
    tokenized = m.split(' ')
    tags = nltk.pos_tag(tokenized)
    
    for tag in tags:
        if tag[1].startswith('JJ'):
            for syn in wordnet.synsets(tag[0]):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        mutated_messages.append(m.replace(tag[0], lm.antonyms()[0].name()))
        if tag[1].startswith('MD'):
            if not tag[0].endswith("n't"):
                mutated_messages.append(m.replace(tag[0], tag[0] +' not').replace(' not not ', ' '))
        if tag[1].startswith('VBN'):
            mutated_messages.append(m.replace(tag[0], 'not '+tag[0]).replace(' not not ', ' '))
    if ' is not ' in m:
        mutated_messages.append(m.replace(' is not ', ' is '))
    elif ' is ' in m:
        mutated_messages.append(m.replace(' is ', ' is not '))
    if ' are not ' in m:
        mutated_messages.append(m.replace(' are not ', ' are '))
    elif ' are ' in m:
        mutated_messages.append(m.replace(' are ', ' are not '))
    if ' was not ' in m:
        mutated_messages.append(m.replace(' was not ', ' was '))
    elif ' was ' in m:
        mutated_messages.append(m.replace(' was ', ' was not '))
    if ' not in ' in m:
        mutated_messages.append(m.replace(' not in ', ' in '))
    elif ' in ' in m:
        mutated_messages.append(m.replace(' in ', ' not in '))
    return mutated_messages