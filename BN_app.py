
# Importing packages

from typing import Counter
import streamlit as st
#import graphviz as graphviz
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11,8)})

from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


# ************** Bayesian Network **************

# probabilities of the final/output node
probs_K = [0.9955156950672646,
 0.004484304932735426,
 0.9983961507618284,
 0.0016038492381716118,
 0.9970760233918129,
 0.0029239766081871343,
 1.0,
 0.0,
 0.9950544015825915,
 0.004945598417408506,
 0.9995475624929306,
 0.00045243750706933604,
 1.0,
 0.0,
 1.0,
 0.0,
 0.9597053726169844,
 0.0402946273830156,
 0.9922504649721017,
 0.007749535027898326,
 0.9412545635579157,
 0.058745436442084305,
 1.0,
 0.0,
 0.966492002206288,
 0.03350799779371208,
 0.9770242373270128,
 0.022975762672987232,
 0.9660678642714571,
 0.033932135728542916,
 0.996742671009772,
 0.003257328990228013,
 0.2987551867219917,
 0.7012448132780082,
 0.9636363636363636,
 0.03636363636363636,
 0.27543424317617865,
 0.7245657568238213,
 0.7272727272727273,
 0.2727272727272727,
 0.297423887587822,
 0.702576112412178,
 0.9888992289284765,
 0.01110077107152353,
 0.4772093023255814,
 0.5227906976744187,
 0.8571428571428571,
 0.14285714285714285,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 0.9964751498061333,
 0.0035248501938667607,
 0.9996275142786193,
 0.00037248572138068043,
 0.9993868792152054,
 0.0006131207847946045,
 1.0,
 0.0,
 0.995049504950495,
 0.0049504950495049506,
 0.9990375360923965,
 0.0009624639076034649,
 0.9781357882623706,
 0.02186421173762946,
 0.9583333333333334,
 0.041666666666666664,
 0.9820506515859356,
 0.017949348414064422,
 0.9944720322308629,
 0.005527967769137075,
 0.9845916795069337,
 0.015408320493066256,
 1.0,
 0.0,
 0.5882352941176471,
 0.4117647058823529,
 0.9868421052631579,
 0.013157894736842105,
 0.4689655172413793,
 0.5310344827586206,
 0.6666666666666666,
 0.3333333333333333,
 0.5966076696165191,
 0.4033923303834808,
 0.9215126650017837,
 0.0784873349982162,
 0.75,
 0.25,
 0.64,
 0.36,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 0.995676500508647,
 0.0043234994913530014,
 0.9993227481375574,
 0.0006772518624426217,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 1.0,
 0.0,
 0.9405405405405406,
 0.05945945945945946,
 1.0,
 0.0,
 0.9788182831661093,
 0.021181716833890748,
 0.9949064051954667,
 0.005093594804533299,
 0.988929889298893,
 0.01107011070110701,
 1.0,
 0.0,
 0.8,
 0.2,
 0.925,
 0.075,
 0.2916666666666667,
 0.7083333333333334,
 0.5,
 0.5,
 0.8488372093023255,
 0.1511627906976744,
 0.9179331306990881,
 0.08206686930091185,
 0.7972972972972973,
 0.20270270270270271,
 0.8,
 0.2]

# Creating the Nodes
A = BbnNode(Variable(0, 'Age', ['<25', '25-50', '>50']),
            [0.091812, 0.682277, 0.225901])
B = BbnNode(Variable(1, 'NoOfPreviousLoansBeforeLoan', ['<5', '>=5']),
            [0.89781, 0.10219])
C = BbnNode(Variable(2, 'Gender', ['0', '1', '2']),
            [0.601021, 0.330772, 0.068207])
D = BbnNode(Variable(3, 'AppliedAmount', ['<1000', '1000-5000', '>5000']),
            [0.288571, 0.573644, 0.137786])
E = BbnNode(Variable(4, 'FreeCash', ['<500', '>=500']),
            [0.930717, 0.069283])
F = BbnNode(Variable(5, 'AmountOfPreviousLoansBeforeLoan', ['<500', '500-5000', '>5000']),
            [0.619135, 0.313937, 0.066928,
             0, 0.3273, 0.6727,
             0.517632, 0.318676, 0.163691,
             0, 0.116155, 0.883845,
             0.521123, 0.298655, 0.180223,
             0, 0.110323, 0.889677])
G = BbnNode(Variable(6, 'Interest', ['<20', '20-50', '>50']),
            [0.245866, 0.605173, 0.148961,
             0.614764, 0.361840, 0.023396,
             0.246667, 0.671259, 0.082074,
             0.483924, 0.502816, 0.013261,
             0.013509, 0.228742, 0.757749,
             0.037736, 0.371069, 0.591195])
H = BbnNode(Variable(7, 'LoanDuration', ['<36', '36-60']),
            [0.198832, 0.801168,
             0.092086, 0.907914,
             0.056527, 0.943473])
I = BbnNode(Variable(8, 'DebtToIncome', ['<15', '>=15']),
            [0.884442, 0.115558,
             0.460702, 0.539298])
J = BbnNode(Variable(9, 'PreviousEarlyRepaymentsBeforeLoan', ['<3000', '>=3000']),
            [0.314174, 0.685826])
K = BbnNode(Variable(10, 'Target', ['0', '1']),
            probs_K)


# Creating the Bayesian Netwrok structure
bbn = Bbn() \
    .add_node(A) \
    .add_node(B) \
    .add_node(C) \
    .add_node(D) \
    .add_node(E) \
    .add_node(F) \
    .add_node(G) \
    .add_node(H) \
    .add_node(I) \
    .add_node(J) \
    .add_node(K) \
    .add_edge(Edge(A, F, EdgeType.DIRECTED)) \
    .add_edge(Edge(B, F, EdgeType.DIRECTED)) \
    .add_edge(Edge(B, G, EdgeType.DIRECTED)) \
    .add_edge(Edge(C, G, EdgeType.DIRECTED)) \
    .add_edge(Edge(D, H, EdgeType.DIRECTED)) \
    .add_edge(Edge(E, I, EdgeType.DIRECTED)) \
    .add_edge(Edge(F, K, EdgeType.DIRECTED)) \
    .add_edge(Edge(G, K, EdgeType.DIRECTED)) \
    .add_edge(Edge(H, K, EdgeType.DIRECTED)) \
    .add_edge(Edge(I, K, EdgeType.DIRECTED)) \
    .add_edge(Edge(J, K, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)


# ************** Front end Interface **************

st.title('Bayesian Network for Credit Risk Analysis of Peer-to-Peer Lending')
st.markdown('This app calculates the probability of loan default using a Bayesian Network.\
         Select the variable inputs below on the left side and the output will be shown on the right side.')

C1, C2 = st.columns(2)



# Defining variables for evidence for respective nodes
global a, b, c, d, e, f, g, h, i, j, k
global ec

# Adding widgets

with C1:
    a = st.slider('Age', 0, 100)
    b = st.selectbox('Number of loans taken before this loan', ('NA', 'Less than 5', '5 or more'))
    c = st.selectbox('Gender', ('NA', 'Male', 'Female', 'No response'))
    d = st.selectbox('Applied loan amount', ('NA', 'Less than $1000', '$1000 to $5000', 'More than $5000'))
    e = st.selectbox('Amount of free cash', ('NA', 'Less than $500', '$500 or more'))
    f = st.selectbox('Amount of previous loans taken before this loan', ('NA', 'Less than $500', '$500 to $5000', 'More than $5000'))
    g = st.selectbox('Interest bracket', ('NA', 'Less than 20%', '20% to 50%', 'More than 50%'))
    h = st.slider('Loan duration (in months)', 0, 60)
    i = st.selectbox('Debt to income ratio', ('NA', 'Less than 15', '15 or more'))
    j = st.selectbox('Amount of previous early repayments before this loan', ('NA', 'Less than $3000', '$3000 or more'))

    if a >= 14 and a < 25:
        a = '<25'
    elif a >= 25 and a <= 50:
        a = '25-50'
    elif a > 50:
        a = '>50'
    else:
        a = 'NA'

    if b == 'Less than 5':
        b = '<5'
    elif b == '5 or more':
        b = '>=5'

    if c == 'Male':
        c = '0'
    elif c == 'Female':
        c = '1'
    elif c == 'No response':
        c = '2'

    if d == 'Less than $1000':
        d = '<1000'
    elif d == '$1000 to $5000':
        d = '1000-5000'
    elif d == 'More than $5000':
        d = '>5000'

    if e == 'Less than $500':
        e = '<500'
    elif e == '$500 or more':
        e = '>=500'

    if f == 'Less than $500':
        f = '<500'
    elif f == '$500 to $5000':
        f = '500-5000'
    elif f == 'More than $5000':
        f = '>5000'

    if g == 'Less than 20%':
        g = '<20'
    elif g == '20% to 50%':
        g = '20-50'
    elif g == 'More than 50%':
        g = '>50'

    if h == 0:
        h = 'NA'
    elif h > 0 and h < 36:
        h = '<36'
    elif h > 36:
        h = '36-60'

    if i == 'Less than 15':
        i = '<15'
    elif i == '15 or more':
        i = '>=15'

    if j == 'Less than $3000':
        j = '<3000'
    elif j == '$3000 or more':
        j = '>=3000'

    def print_target_prob():
        potential = join_tree.get_bbn_potential(K)
        C2.write('OUTPUT:')
        C2.write('Probability of loan default')
        C2.write()
        C2.write(potential, )
        C2.write('-' * 25)

    def add_evidence(ev, nod_id, cat, val):
        # 0 - 0.96803
        # 1 - 0.03197
        # ev = 
        join_tree.set_observation(EvidenceBuilder() \
            .with_node(join_tree.get_bbn_node(nod_id)) \
            .with_evidence(cat, val) \
            .build())

    
    def calc():
        vl = [a, b, c, d, e, f, g, h, i , j]
        for v in range(len(vl)):
            if vl[v] != 'NA':
                add_evidence(0, v, vl[v], 1)
        print_target_prob()
    
    


# Buttons and Bayesian Network graph

with C2:
    st.button('Calculate', on_click = calc, help='Click to calculate the probability of default.')

    st.graphviz_chart('''
        digraph {
            Age -> Amount of previous Loans taken before this loan
            Number of loans taken before this loan -> Amount of previous Loans taken before this loan
            Amount of previous Loans taken before this loan -> Interest bracket
            Gender -> Interest bracket
            Applied loan amount -> Loan duration (in months)
            Amount of Free cash -> Debt to income ratio
            Amount of previous Loans taken before this loan -> Probability of loan default
            Interest bracket -> Probability of loan default
            Loan duration (in months) -> Probability of loan default
            Debt to income ratio -> Probability of loan default
            Amount of previous early repayments before this loan -> Probability of loan default
        }
    ''')

    n, _ = bbn.to_nx_graph()
    labels = {0: 'Age\n\n\n',
              1: 'Number of loans taken\n before this loan\n\n\n',
              2: 'Gender\n\n\n',
              3: 'Applied loan amount\n\n\n',
              4: 'Amount of Free cash\n\n\n',
              5: '\n\n\nAmount of previous Loans\n taken before this loan',
              6: '\n\n\nInterest bracket',
              7: '\n\n\nLoan duration\n (in months)',
              8: '\n\n\nDebt to income ratio',
              9: 'Amount of previous\n early repayments\n before this loan\n\n\n\n',
              10: '\n\n\nProbability of\n loan default'}

    pos = {0: (-2, 3), 1: (-1, 2.95), 2: (0, 2.5), 3: (1, 3), 4: (1.75, 2.5),
           5: (-1.75, 1.5), 6: (-1, 2), 7: (0, 2), 8: (1, 2), 9: (1.85, 1.5),
           10: (0, 1)}

    fig, ax = plt.subplots()
    ax = nx.draw(n, with_labels=True, labels=labels, pos=pos, font_size=14, alpha=1)
    st.pyplot(fig)

#st.button('Calculate', on_click = calc(), help='Click to calculate the probability of default.')


st.write('-' * 100)
st.write('Project by:')
st.write('Pranav Mahajan')
st.write('Aditya Ramakrishnan')
st.write('Yash Choudhari')