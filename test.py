import ply.lex as lex
import ply.yacc as yacc
import sys

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
t_load = r'load'
t_ignore = r''

def t_float :
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_int :
    r'\d+'
    t.value = int(t.value)
    return t

def t_string :
    r'[a-zA-z]+'
    t.type = 'string'
    return t

def t_filepath :
    r'[a-zA-Z]+\.csv'
    t.type = 'filepath'
    return t

def t_error(t) :
    print('Illegal type ')
    t.lexer.skip(1)

