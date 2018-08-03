import ply.lex as lex
import ply.yacc as yacc
import sys
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from pandas import Series , DataFrame as df



# Create a list to hold all of the token names
tokens = [

    'int',
    'float',
    'load',
    'split',
    'filepath',
    'input',
    'output',
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
    'adam',
    'tanh',
    'linear',
    'adagrad',
    'rmsprop',
    'sgd'

]
t_load = r'(load)'
t_split = r'(split)'
t_input = r'(input)'
t_output = r'(output)'
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
t_tanh = r'(tanh)'
t_linear = r'(linear)'
t_adagrad = r'(adagrad)'
t_rmsprop = r'(rmsprop)'
t_sgd = r'(sgd)'
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

def t_error(t) :
    print('Illegal type ')
    t.lexer.skip(1)

lexer = lex.lex()
'''
lexer.input("load read.csv")
while True:
    tok = lexer.token()
    if not tok:
        break
    print(tok)
'''
'''while True:
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
'''
data = df()
X = df()
Y= df()
model = Sequential()
hid_no = 0
input_dim =0
# --------------------------------------------- YACCC ------------
def p_start(p):
    '''
    start : LoadFile Split Use Addl Addl2 Compile Fit Evaluate
    '''
    p[0] = (p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    #print (p[0])

def p_LoadFile(p) :
    '''
    LoadFile : load filepath
    '''
    p[0] = (p[1], p[2])
    global data
    data = pd.read_csv(p[2])
    print(data.head())

def p_Split(p) :
    '''
    Split : split splitinput splitoutput
    '''
    p[0] = (p[1], p[2])

def p_splitinput(p) :
    '''
    splitinput : input int int
    '''
    p[0] = (p[1], p[2],p[3])
    global data
    global X
    #global Y
    #print(type(p[2]),p[3])
    #print(data.head())
    X = data.iloc[:,p[2]:p[3]]
    #print(X)

def p_splitoutput(p) :
    '''
    splitoutput : output int
    '''
    p[0] = (p[1], p[2])
    global data
    #global X
    global Y
    #print(type(p[2]),p[3])
    Y = data.iloc[:,p[2]]
    #print(Y)

def p_Use(p) :
    '''
    Use : use TON
    '''
    p[0] = (p[1], p[2])
    #print('shghsag')


def p_TON(p) :
    '''
    TON : sequential
    '''
    p[0] = (p[1])
    model = Sequential()
    print(model)

def p_Addl(p) :
    '''
    Addl : add_layer int int Act
    '''
    p[0] = (p[1], p[2], p[3], p[4])
    if p[4]=='relu':
        model.add(Dense(units=p[3], input_shape = (p[2], ), kernel_initializer = 'uniform', activation='relu'))
    elif p[4]=='tanh':
        model.add(Dense(units=p[3], input_shape = (p[2], ), kernel_initializer = 'uniform', activation='tanh'))
    elif p[4]=='linear':
        model.add(Dense(units=p[3], input_shape = (p[2], ), kernel_initializer = 'uniform', activation='linear'))
    else:
        model.add(Dense(units=p[3], input_shape = (p[2], ), kernel_initializer = 'uniform', activation = 'sigmoid'))

def p_Addl2(p) :
    '''
    Addl2 : add_layer int Act Addl2
    '''
    p[0] = (p[1], p[2], p[3], p[4])
    #print(p[2])
    if p[1]!=None:
        if p[3]=='relu':
            model.add(Dense(units=p[2], kernel_initializer = 'uniform', activation='relu'))
        elif p[3]=='tanh':
            model.add(Dense(units=p[3], kernel_initializer = 'uniform', activation='tanh'))
        elif p[3]=='linear':
            model.add(Dense(units=p[3], kernel_initializer = 'uniform', activation='linear'))
        else:
            model.add(Dense(units=p[2], kernel_initializer = 'uniform', activation = 'sigmoid'))

def p_Addl2empty(p) :
    '''
    Addl2 :
    '''
   # print('vaibhav')
    p[0]= None

def p_Act(p) :
    '''
    Act : relu
        | sigmoid
        | tanh
        | linear
    '''
    p[0]=(p[1]) 
    #print(" sckj")

def p_Compile(p) :
    '''
    Compile : compile Optimizer
    '''
    p[0] = (p[1], p[2])
    global model
    model.compile(loss = 'binary_crossentropy', optimizer = p[2], metrics = ['accuracy'])

def p_Optimizer(p) :
    '''
    Optimizer : adam
              | adagrad
              | sgd
              | rmsprop
    '''
    p[0] = (p[1])

def p_Fit(p) :
    '''
    Fit : fit epoch BatchSize
    '''
    p[0] = (p[1], p[2],p[3])
    global X
    global Y
    global model
    model.fit(X.values,Y.values,epochs=p[2], batch_size=p[3], verbose = 1)

def p_epoch(p) :
    '''
    epoch : int
    '''
    p[0] = (p[1])

def p_BatchSize(p) :
    '''
    BatchSize : int
    '''
    p[0] = (p[1])

def p_Evaluate(p) :
    '''
    Evaluate : evaluate
    '''
    p[0] = (p[1])
    global model
    global X
    global Y
    print("Accuracy",model.evaluate(X.values,Y.values)[1]*100,"%")
'''
def p_error(p):
    print("Syntax error found!")
'''
parser = yacc.yacc()
while True :
    try :
        s=""
        s2=""
        while(s!='end'):
            s2 = s2+" "+s
            s = input (">> ") 
    except EOFError :
        break
    print(s2)
    parser.parse(s2)







#load diabetes.csv split input 0 8 output 8 use sequential add_layer 8 12 relu add_layer 1 sigmoid add_layer 8 relu compile adam fit 50 10 evaluate
'''
load diabetes.csv
split input 0 8 output 8
use sequential
add_layer 8 12 relu
add_layer 1 sigmoid
add_layer 8 relu
compile adam
fit 50 10
evaluate
end
'''