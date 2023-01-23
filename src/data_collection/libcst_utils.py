import libcst as cst

# Find print and log statements
class FindCall(cst.CSTVisitor):
    
    prints = []
    raises = []
    other_calls = []
    
    def __init__(self):
        self.prints = []
        self.other_calls = []
        self.raises = []
        
    def visit_Call(self, node: cst.Call):
        try:
            if node.func.value == 'print' or 'log' in str(node.func.value):
                self.prints.append(node)
            else:
                self.other_calls.append(node)
        except Exception as e:
            pass
            #print("CALL DOES NOT HAVE ATTR VALUE", cst.Module([node]).code)
    
class FindComment(cst.CSTVisitor):
    """
    Visitor class to find comments in code

    Attributes:
        comments (list): a list of the found comments in the visited code
    """
    
    def __init__(self):
        """
        Constructor of the class, initializes the attribute comments with and empty list
        """
        self.comments = []

    def visit_Comment(self, node: cst.Comment):
        """
        Visit each node of type Comment in the ast and add the found comments to the list comments

        Parameters:
            node (cst.Comment): any node of type Comment found in the visited ast

        """
        self.comments.append(node)



class FindElse(cst.CSTVisitor):
    """
    The FindElse class is a CSTVisitor subclass that visits Else nodes in a syntax tree and adds them to a list.

    Attributes:
        else_list (List[cst.Else]): a list that contains all Else nodes visited by the visitor.

    Methods:
        visit_Else(self, node: cst.Else): visits an Else node and adds it to the else_list.
    """
    def __init__(self):
        """
        Initializes the FindElse class and creates an empty list to store Else nodes.
        """
        self.else_list = []
    
    def visit_Else(self, node: cst.Else):
        """
        Visits an Else node in the syntax tree and adds it to the else_list attribute.

        Parameters:
            node (cst.Else): the Else node to be visited and added to the else_list.
        """
        self.else_list.append(node)    
        

class FindString(cst.CSTVisitor):
    """
    The FindString class is a CSTVisitor subclass that visits SimpleString nodes in a syntax tree and adds them to a list.

    Attributes:
        strings (List[cst.SimpleString]): a list that contains all SimpleString nodes visited by the visitor.

    Methods:
        visit_SimpleString(self, node: cst.SimpleString): visits a SimpleString node and adds it to the strings attribute.
    """
    
    def __init__(self):
        """
        init(self)

        Initializes the FindString class and creates an empty list to store SimpleString nodes.
        """
        self.strings = []
        
    def visit_SimpleString(self, node: cst.SimpleString):
        """
        Visits a SimpleString node in the syntax tree and adds it to the strings attribute.

        Parameters:
            node (cst.SimpleString): the SimpleString node to be visited and added to the strings attribute.
        """
        self.strings.append(node)
        
        
class FindPrint(cst.CSTVisitor):
    """
    The FindPrint class is a CSTVisitor subclass that visits Call nodes in a syntax tree and separates them into three different lists: prints, other_calls, and prints_detailed.

    Attributes:
        prints (List[cst.Call]): a list that contains all Call nodes with 'print' as the function being called.
        other_calls (List[cst.Call]): a list that contains all Call nodes that are not calling 'print'
        prints_detailed (List[Tuple[List[cst.Arg], cst.Call]]): a list that contains all tuple of argument list and the Call node with 'print' as the function being called.

    Methods:
        visit_Call(self, node: cst.Call): visits a Call node, checks if the function being called is 'print', and adds it to the appropriate list.
    """
    def __init__(self):
        """
        Initializes the FindPrint class and creates three empty list to store Call nodes: prints, other_calls, and prints_detailed.
        """
        self.prints = []
        self.other_calls = []
        self.prints_detailed = []
        
    def visit_Call(self, node: cst.Call):
        """
        Visits a Call node in the syntax tree, checks if the function being called is 'print', and adds it to the appropriate list: prints, other_calls, and prints_detailed.

        Parameters:
            node (cst.Call): the Call node to be visited and added to the appropriate list.
        """
        try:
            if node.func.value == 'print':
                self.prints.append(node)
                self.prints_detailed.append((node.args))
            else:
                self.other_calls.append(node)
        except:
            pass
                        
class FindRaise(cst.CSTVisitor):
    """
    The FindRaise class is a CSTVisitor subclass that visits Raise nodes in a syntax tree and adds them to a list.

    Attributes:
        raises (List[cst.Raise]): a list that contains all Raise nodes visited by the visitor.

    Methods:
        init(self): Initializes the FindRaise class and creates an empty list to store Raise nodes.
        visit_Raise(self, node: cst.Raise): visits a Raise node and adds it to the raises attribute.
    """
    def __init__(self):
        self.raises = []
        
    def visit_Raise(self, node: cst.Raise):
        """
        Visits a Raise node in the syntax tree and adds it to the raises attribute.

        Parameters:
            node (cst.Raise): the Raise node to be visited and added to the raises attribute.
        """
        self.raises.append(node)
        
    
class FindIf(cst.CSTVisitor):
    """
    The FindIf class is a CSTVisitor subclass that visits If nodes in a syntax tree and adds them to a list and visited_nodes list.

    Attributes:
        if_stmts (List[Tuple[List[str], List[str]]]): a list that contains all If nodes and their condition and else statement visited by the visitor.
        visited_nodes (List[cst.If]): a list that contains all If nodes visited by the visitor.

    Methods:
        visit_If(self, node: cst.If): visits a If node, checks if the if statement is nested and adds it to the if_stmts attribute and visited_nodes.
    """
    def __init__(self):
        self.if_stmts = []
        self.visited_nodes = []
        
    def visit_If(self, node: cst.If):
        """
        visit_If(self, node: cst.If)

        Visits a If node in the syntax tree, checks if the if statement is nested and adds it to the if_stmts attribute and visited_nodes.

        Parameters:
        node (cst.If): the If node to be visited and added to the if_stmts attribute and visited_nodes.
        """
        else_if = '' 
        node1 = node
        condition = ''
        while type(node1) == cst._nodes.statement.If:
            if node1 not in self.visited_nodes:
                self.if_stmts.append(
                    (["if", cst.Module([node1.test]).code+condition, None, else_if],["if", node1, None, node1.orelse])
                )
                self.visited_nodes.append(node1)
                condition = condition + ' and not (' + cst.Module([node1.test]).code + ')'
                node1 = node1.orelse
            else:
                break
        if node1!=None:
            self.if_stmts.append(
            (["if", condition[4:], None, else_if],["if", node1, None, None])
            )
        
class RemoveIf(cst.CSTVisitor):
    """
    The RemoveIf class is a CSTVisitor subclass that visits If nodes in a syntax tree and removes it.

    Attributes:
        to_remove (List[cst.Node]): a list of nodes to remove from the If statement's body.
        exceptions (List[Exception]): a list of exceptions that occur during the removal process.

    Methods:
        visit_If(self, node: cst.If): visits a If node and removes the nodes passed in the to_remove list from the If's body, and records any exceptions that occur.
    """
    def __init__(self, to_remove):
        self.to_remove = to_remove
        self.exceptions = []
    def visit_If(self, node: cst.If):
        """
        visit_If(self, node: cst.If)

        Visits a If node and removes the nodes passed in the to_remove list from the If's body, and records any exceptions that occur.

        Parameters:
            node (cst.If): the If node to be visited and have the nodes removed from its body.
        """
        for tr in self.to_remove:
            try:
                node.body.body.remove(tr)
            except Exception as e:
                self.exceptions.append(e)
        

def get_simple_ifs(finder):
    """
    get_simple_ifs(finder: FindIf) -> List[Tuple[List[str], List[str]]]

    This function takes a FindIf object and returns a list of If statements with their nested If statements removed.

    Parameters:
        finder (FindIf): FindIf object that contains the list of If statements to be processed.

    Returns:
        List[Tuple[List[str], List[str]]]: A list of If statements with their nested If statements removed.
    """
    new_ifs = []
    for if_stmt in finder.if_stmts:
        finder1 = FindIf()
        _ = if_stmt[1][1].visit(finder1)
        remover = RemoveIf([s[1][1] for s in finder1.if_stmts])
        new_ifs.append((if_stmt[0], [if_stmt[1][0], if_stmt[1][1].visit(remover), if_stmt[1][2], if_stmt[1][3]]))
    return new_ifs


def node_to_code(node):
    """
    node_to_code(node: cst.Node) -> str

    This function takes a CST Node and returns the code representation of the node.

    Parameters:
        node (cst.Node): The CST Node that should be converted to code

    Returns:
        str: The code representation of the input node

    """
    return cst.Module([node]).code


def remove_else(if_stmt):
    """
    remove_else(if_stmt: Tuple[List[str], List[str]]) -> Union[str, None]

    This function takes an if statement represented as a tuple of lists, removes the else statement and returns the if statement as a string.

    Parameters:
        if_stmt (Tuple[List[str], List[str]]): The if statement that should have its else statement removed.

    Returns:
        Union[str, None]: The if statement as a string, with the else statement removed or None if the input if statement has no condition.
    """
    if_only = []
    code_s = node_to_code(if_stmt[1][1].body).split('\n')
    if_only.append('if '+if_stmt[0][1]+':')
    if if_stmt[0][1] == '':
        return None
    p_i = -1
    for l in code_s:
        i = 0
        for c in l:
            if c==' ':
                i += 1
            else:
                break
        if i < p_i:
            return '\n'.join(if_only)
        else:
            if_only.append(l)
            p_i = i

class FindFunctionDef(cst.CSTVisitor):
    """
    A class that inherits from the cst.CSTVisitor class and is used to find and store 
    function definitions within a given codebase.

    Attributes:
    functions (list): A list of tuples containing the name and code of the functions found
    """
    def __init__(self):
        self.functions = []
        
    def visit_FunctionDef(self, node):
        """
        Visit a function definition node in the CST and store its name and the code for the function.

        Parameters:
            node (cst.FunctionDef): The function definition node to visit.

        Returns:
            None
        """
        self.functions.append((node.name.value, cst.Module([node]).code))


def extract_batch_functions(paths):
    """
    Extracts all functions from a list of file paths and returns a list of tuples
    containing the file path and the functions found in each file.

    Parameters:
        paths (List[str]): A list of file paths to extract functions from.

    Returns:
        List[Tuple[str, List[Tuple[str, str]]]]: A list of tuples containing the file path and a list of tuples
        containing the name and code of the functions found in the file.
    """
    functions_list = []
    for f_path in paths:
        try:
            with open(f_path) as fp:
                code = fp.read()
        except Exception as e:
            print(e)
            continue
        try:
            function_finder = FindFunctionDef()
            tree = cst.parse_module(code)
            _ = tree.visit(function_finder)

            functions_list.append((f_path, function_finder.functions))

        except Exception as e:
            pass
        
    return functions_list

def extract_ifs(code):
    """
    Extracts simple if statements from a given code.

    Parameters:
        code (str): The code to extract if statements from.

    Returns:
        List[str]: A list of simple if statements in the given code.
    """
    finder = FindIf()
    code = code.split('\n')
    
    try:

        tree = cst.parse_module('\n'.join(code))
        _ = tree.visit(finder)
        ifs = get_simple_ifs(finder)
    except Exception as e:
        ifs = []
    return ifs  

class FunctionExtractor(cst.CSTVisitor):
    def __init__(self):
        self.functions = []
    
    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.functions.append(cst.Module([node]).code)