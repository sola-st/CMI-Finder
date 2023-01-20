import libcst as cst

class FindIdentifiers(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
        
    def visit_Name(self, node: cst.Name):
        if node.value in self.names_dict:
            pass
        else:
            self.names_dict[node.value] = 'IDTF'+str(self.id_count)
            self.id_count += 1
            
class FindInteger(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
        
    def visit_Integer(self, node: cst.Integer):
        if node.value in self.names_dict:
            pass
        else:
            self.names_dict[node.value] = 'INTG'+str(self.id_count)
            self.id_count += 1

class FindFloat(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
        
    def visit_Float(self, node: cst.Float):
        if node.value in self.names_dict:
            pass
        else:
            self.names_dict[node.value] = 'FLT'+str(self.id_count)
            self.id_count += 1
            
class FindString(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
    def visit_SimpleString(self, node: cst.SimpleString):
        if node.value in self.names_dict:
            pass
        else:
            self.names_dict[node.value] = 'STRNG' +str(self.id_count)
            self.id_count += 1
            
class FindList(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
    def visit_List(self, node: cst.List):
        self.names_dict[str(cst.Module([node]).code)] = 'LST' +str(self.id_count)
        self.id_count += 1
        
class FindTuple(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
    def visit_Tuple(self, node: cst.Tuple):
        self.names_dict[str(cst.Module([node]).code)] = 'TPL' +str(self.id_count)
        self.id_count += 1
        
        
class FindSet(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
    def visit_Set(self, node: cst.Set):
        self.names_dict[str(cst.Module([node]).code)] = 'SET' +str(self.id_count)
        self.id_count += 1
        
class FindDict(cst.CSTVisitor):
    
    def __init__(self):
        self.id_count = 0
        self.names_dict = {}
        
    def visit_Dict(self, node: cst.Dict):
        self.names_dict[str(cst.Module([node]).code)] = 'DCN' +str(self.id_count)
        self.id_count += 1

