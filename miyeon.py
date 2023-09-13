import pandas as pd
import tensorflow as tf

OUT_LABELS = ['01-A (mass number)','01-Z (atomic number)','01-q (ion charge)']
IN_LABELS = ['01-TKE [MeV]','01-Time from Target [ns]','01-Brho [T*m]',
          '02-Energy Loss (MeV)','03-Energy Loss (MeV)','04-Energy Loss (MeV)']

MYOPT = tf.keras.optimizers.Adam()
MYLOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def pull(file):
    raw = pd.read_csv(file,index_col=0)
    #'01-Momentum [GeV/c]','02-Momentum [GeV/c]','03-Momentum [GeV/c]','04-Momentum [GeV/c]'
    #We could add these later if we want

    listy=[]
    for o in OUT_LABELS:
        listy.append(raw.pop(o))

    outputs = pd.DataFrame(listy)
    outputs=outputs.T

    listy=[]
    for i in IN_LABELS:
        listy.append(raw.pop(i))

    inputs = pd.DataFrame(listy)
    inputs=inputs.T
   
    return(inputs,outputs)

def normNsplit(inputs,outputs):
    ###Normalization is an option (Should we do it for output labels?)
    # for o in OUT_LABELS:
    #     outputs[o] = outputs[o]/outputs[o].max()

    for i in IN_LABELS:
        inputs[i] = inputs[i]/inputs[i].max()
       
    ###Split the data
    train_in = inputs[0:750].to_numpy()
    train_out = outputs[0:750].to_numpy()

    test_in = inputs[750:1000].to_numpy()
    test_out = outputs[750:1000].to_numpy()

    type(train_in),train_in.shape,type(train_out),train_out.shape
   
    return(train_in,train_out,test_in,test_out)

def maker(out):
    if out == 'A':
        logits = 20
    elif out == 'Z' or out == 'Q':
        logits = 118
   
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(logits)
    ])
   
    model.compile(optimizer=MYOPT,loss=MYLOSS,metrics=['accuracy'])
   
    return model
