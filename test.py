import ply.lex as lex
import ply.yacc as yacc
import sys

#---------------------------------------LEX-part--------------------------------

# Create a list to hold all of the token names
tokens = [

    'int',
    'float',
    'string',
    'load',
    'split',
    'filepath',
    'input',
    'output',
    'index',
    'train',
    'use',
    'sequential',
    'test',
    'add_layer',
    'compile',
    'fit',
    'evaluate',
    'relu',
    'sigmoid',
    'adam'

]
t_load = r'(load)'
t_split = r'(split)'
t_input = r'(input)'
#t_ouptut = r'(output)'
t_train = r'(train)'
t_use = r'(use)'
t_sequential = r'(sequential)'
t_test = r'(test)'
t_add_layer = r'(add_layer)'
t_compile = r'(compile)'
t_fit = r'(fit)'
t_evaluate = r'(evaluate)'
t_relu = r'(relu)'
t_sigmoid = r'(sigmoid)'
t_adam = r'(adam)'
t_ignore = r' '

def t_float(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_int(t):
    r'\d+'
    t.value = int(t.value)
    return t


'''def t_load(t):
    r'(load)'
    t.type = 'load'
    return t
'''
def t_filepath(t) :
    #r'^(.+)\/([^\/]+)$'
    r'[a-zA-z]+\.csv'
    t.type = 'filepath'
    return t

'''def t_string(t):
    r'[a-zA-z]+'
    t.type = 'string'
    return t
'''
def t_index(t) :
    r'\[[0-9]+\]|\[[0-9]+\:[0-9]+\]'
    t.type = 'index'
    return t

def t_error(t) :
    print('Illegal type ')
    t.lexer.skip(1)

#lexer = lex.lex()
'''
lexer.input("load read.csv")

while True:
    tok = lexer.token()
    if not tok:
        break
    print(tok)
'''
while True:
    lexer = lex.lex()
    try:
        s = input('>> ')
    except EOFError:
        break
    print (s)
    lexer.input(s)
    while True :
        tok = lexer.token()
        if not tok:
            break
        print(tok)




#---------------------------------------LEX-part-END-------------------------------


#---------------------------------------YACC-part--------------------------------


#todo


#---------------------------------------YACC-part-END-------------------------------
