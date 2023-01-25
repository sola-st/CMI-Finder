from nltk.tokenize import word_tokenize

rename_globals = {
        'first': '1',
        'second': '2',
        'third': '3',
        'fourth': '4',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'less than': '<',
        'smaller than': '<',
        'lower than': '<',
        'more than': '>',
        'bigger than': '>',
        'greater than': '>',
        'larger than': '>',
        'higher than': '>',
        'equal to': '==',
        'equal': '==',
        'greater': '>',
        'bigger': '>',
        'smaller': '<',
        'larger': '>',
        'higher': '>',
        'more': '>',
        'lower': '<',
        'multiple': '> 1',
        'more': '>',
        'less': '<',
        '< or =' :'<=',
        '> or =': '>=',
        '= or <': '<=',
        '= or >': '>=',
        'not empty': '> 0',
        'empty': '0',
        'not contain': '0',
        'contains no': '0',
        'blank': '0',
        'no ': '0 ',
        'the only': '1',
        'too many': '>',
        'too few': '<',
        'at least': '>=',
        'at most': '<='
    }


special_embedding = {}
    
def tokenize_if_block(code):
    try:
        return word_tokenize(code.replace('`', '').replace("'", ''))
    except Exception as e:
        print(e)

def vectorize_if_block(tokens, model):
    vector = []
    for t in tokens:
        if t in special_embedding:
            vector.append(model.wv[special_embedding[t]])
        else:
            vector.append(model.wv[t])
    return vector


from tokenize import tokenize
from io import BytesIO
def tokenize_python(code):
    
    g = tokenize(BytesIO(code.encode('utf-8')).readline)
    try:
        tokens = [c[1] for c in g if c[1]!='' and c[1]!='\n'][1:]
    except:
        tokens = tokenize_if_block(code)
    
    clean_tokens = []
    
    for t in tokens:
        if ' ' in t:
            for rg in rename_globals:
                t = t.replace(rg, rename_globals[rg])
            clean_tokens += tokenize_if_block(t.replace('"', '').replace("'", ''))
        else:
            clean_tokens.append(t)
    
    return clean_tokens


def tokenize_triplets(triplets):
    return [
            (tokenize_python(t[0]), tokenize_python(t[1]), tokenize_python(t[2])) for t in triplets
    ]