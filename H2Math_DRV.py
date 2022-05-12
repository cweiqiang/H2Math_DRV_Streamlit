import streamlit as st
import pandas as pd
import numpy as np
import sympy
from sympy import sin, cos, tan, exp, log, symbols, lambdify, sympify, Float
from fractions import Fraction
import matplotlib.pyplot as plt

## Section 0
st.title ('Discrete Random Variables - Visualisation and Analysis')
st.markdown( 'This web application allows you to key in the *probability density function* (p.d.f) of any *discrete random variable* (d.r.v) $X$, for instance:') 

df1 = pd.DataFrame({'x':[1, 2], 'P(X=x)':[1/2, 1/2]})
st.write(df1.T.to_html(index=True,header=False), unsafe_allow_html=True)
st.markdown('You will then be automatically generated a statistical dashboard for visualisation and analysis!')

st.markdown('This statisical dashboard includes:')

latext = r'''
- Plot for the *probability density function* (p.d.f) of $X$:  $f(x)=P(X=x)$ 
- Plot for the *probability cumulative function* (c.d.f) of $X$: $F(x)=P(X \leq x)$ 
- *Expectation* of $X$: $\mu=E(X)=\sum_{x} xP(X=x)$
- *Variance* of $X$: $\sigma^{2}=Var(X)=E\left[ (X-\mu)^{2} \right]$, where $\sigma$ is the standard deviation (S.D.)
- *Mode* of $X$: the integer $r$ for which $P(X=r)$ is the largest
- *Median* of $X$: the largest integer $Q_{2}$ for which $F(Q_{2})\leq 0.5$ 
- *1st and 3rd Quartiles*: largest integers $Q_{1},Q_{3}$ such that $F(Q_{1})\leq 0.25$ and $F(Q_{3})\leq 0.75$
- *Interquartile Range* (IQR): $IQR=Q_{3}-Q_{1}$, and is also used as a measure for the spread of values, similar to  standard deviation.
'''
st.write(latext)
st.markdown('You can also apply any arbitary function $g$ on $X$ to get another d.r.v. to experiment.')
st.markdown('**Scroll down and have fun exploring**!')

## Section 1
st.header("Section 1: Input values of $x, p$ for D.R.V.")
raw_input = st.text_input("Please key in all possible values for x:")
   # Validation process 
if  len (raw_input) < 1 :
    st.warning ('The possible values for x keyed in must be space-separated, e.g. 1 2 3')
    # Stop processing if the conditions are not met
    st.stop ()
str_arr = raw_input.split(' ')
X = np.array([int(num) for num in str_arr])
st.write("The input list of values for x is:")
df2 = pd.DataFrame(X, columns=['x'])
st.write(df2.T.to_html(index=True,header=False), unsafe_allow_html=True)

prob_input=st.text_input("Input the probabilities values for p=P(X=x):")
str_arr2 = prob_input.split(' ') #str_arr2 is a Python list
if len(str_arr2) != len(X) or len(str_arr2) < 1:
    st.warning ('Ensure the number of probability values p is equal to the number of input values for x above!')
    st.warning ('The probabilities values keyed in for p must be space-separated, e.g. 1/4 1/2 1/4')
    st.warning ('The probabilities values must be in fractional form or decimal form, e.g. 1/4 1/2 1/4 or 0.25 0.5 0.25 or 1/4 0.5 1/4')
    # Stop processing if the conditions are not met
    st.stop()

p=[]
if str_arr2:
    prob= sympify(str_arr2)
    for num in prob:
        if sympify(num.is_Float):
            p.append(float(Float(num)))
        if sympify(num.is_Rational):
            p.append(float(Fraction(num)))
p=np.array(p)
epsilon=1e-7
if np.abs(1-np.sum(p))>epsilon:
    st.warning ('The probabilities values should add up to 1 with error less than $10^{-7}$, please check and re-enter the values for p above')
    st.stop()
# st.write(p)
df3=pd.DataFrame({'x': X, 'P(X=x)': p}, columns=['x', 'P(X=x)'])
st.write(df3.T.to_html(index=True,header=False), unsafe_allow_html=True)

## Section 2: Data Visualisation and Analysis
st.header("Section 2: Data Visualisation and Analysis")
st.write(" The following is a statisical dashboard for visualisation and analysis.")

Expectation=np.sum(np.multiply(X,p))
Variance=np.sum(np.multiply(np.power(X,2),p))-Expectation**2
Std_Dev=np.power(Variance,0.5)

mode_index=np.argmax(p)
Mode=X[mode_index]

# Median (Q2)
median_index=np.argmax(np.reciprocal(0.5-np.cumsum(p)+epsilon))
Median=X[median_index]

# Quartile1 (Q1)
Q1_index=np.argmax(np.reciprocal(0.25-np.cumsum(p)+epsilon))
Quartile1=X[Q1_index]
# Quartile3 (Q3)
Q3_index=np.argmax(np.reciprocal(0.75-np.cumsum(p)+epsilon))
Quartile3=X[Q3_index]
IQR=Quartile3-Quartile1

stats_array=np.array([Expectation,Variance,Std_Dev,Mode,Quartile1,Median,Quartile3,IQR]).T
stats_col = ['E(X)','Var(X)', 'S.D.', 'Mode', 'Q1', 'Q2 (Median)', 'Q3', 'IQR']
df_stats = pd.DataFrame(stats_array, index=stats_col, columns=['Statistics'])
st.write(df_stats.T.to_html(index=True), unsafe_allow_html=True)

#Barplots of p.d.f and c.d.f. of random variable X
fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.bar(X, p, color='orange', linewidth=2, tick_label=X)
ax1.set_title('Plot for p.d.f of X')
ax2.bar(X, np.cumsum(p),color='green', linewidth=1, tick_label=X )
ax2.set_title('Plot for c.d.f of X')
st.pyplot(fig)

st.write("**Q1**: Comparing the expectation, median and mode of X, what could we conclude about the skewness of this distribution (left or right skewed or symmetrical)?")

latextQ1 = r'''
If the distribution is not symmetrical, then the mean, mode and median can be used to figure out if you have a left or right-skewed distribution.
- If the mean is greater than the mode, the distribution is right-skewed (long-right tailed).
- If the mean is less than the mode, the distribution is left-skewed (long-left tailed).
- If the mean is greater than the median, the distribution is right-skewed (long-right tailed).
- If the mean is less than the median, the distribution is left-skewed (long-left tailed).
'''
st.text_input("Enter your answer for Q1 here.")
if st.button('Click here to compare your answer against the explanation of Q1'):
     st.write(latextQ1)




st.write("**Q2**: For this distribution, is standard deviation (S.D.) or interquartile range (IQR) a more appropriate measure for the spread of values?")

latextQ2 = r'''
Both metrics measure the spread of values in a dataset.
However, the interquartile range and standard deviation have the following key difference:

- The interquartile range (IQR) is not affected by extreme outliers. For example, an extremely small or extremely large value in a dataset will not affect the calculation of the IQR because the IQR only uses the values at the 25th percentile and 75th percentile of the dataset.

- The standard deviation is affected by extreme outliers. For example, an extremely large value in a dataset will cause the standard deviation to be much larger since the standard deviation uses every single value in a dataset in its formula.

You should use the interquartile range to measure the spread of values in a dataset when there are extreme outliers present.
'''

st.text_input("Enter your answer for Q2 here.")
if st.button('Click here to compare your answer against explanation of Q2'):
     st.write(latextQ2)

## Section 3: Transfoming X with function g(X) [Optional]
st.header("Section 3: Transfoming D.R.V. with functions")
expr = st.text_input('Input the expression of transformation function g(x) here: ')
if  len(expr) < 1 :
    st.warning('Examples of expression for g(x)=2*x+1,x**2, exp(x), sin(x), cos(x), log(x)')
    st.warning('Please ensure the expression input for g(x) is in terms of x, not X! Lettercase sensitive.')
    # Stop processing if the conditions are not met
    st.stop()
if expr:
    func = sympify(expr)
    st.write(func)


x = symbols('x')
g = lambdify(x, func, 'numpy')
st.write('**Please ensure the expression input for $g(x)$ is in terms of $x$, not $X$! It is lettercase sensitive.**')
st.write('**Examples of $g(x)$ for input**: 2*x+1, x**2, exp(x), sin(x), cos(x), log(x)')
st.write(' The above examples correspond to  $2x+1$, $x^{2}$, $e^{x}$, $\sin(x)$, $\cos(x)$, $\ln(x)$.')
st.write("The P.D.F of $g(X)$ is given as follows:")
df4 = pd.DataFrame({'x': X, 'g(x)': g(X), 'P(X=x)': p}, columns=['x','g(x)', 'P(X=x)'])

st.write(df4.T.to_html(index=True,header=False), unsafe_allow_html=True)



Expectation_g=np.sum(np.multiply(g(X),p))
Variance_g=np.sum(np.multiply(np.power(g(X),2),p))-Expectation_g**2
Std_Dev_g=np.power(Variance_g,0.5)


stats_array_g=np.array([Expectation_g,Variance_g,Std_Dev_g]).T
stats_col_g = ['E(g(X))','Var(g(X))', 'S.D.']
df_stats_g = pd.DataFrame(stats_array_g, index=stats_col_g, columns=['Statistics'])
st.write("Please view and compare the expectations and variances for $g(X)$ and $X$.")
st.write(df_stats_g.T.to_html(index=True), unsafe_allow_html=True)

stats_array2=np.array([Expectation,Variance,Std_Dev]).T
stats_col2 = ['E(X)','Var(X)', 'S.D']
df_stats2 = pd.DataFrame(stats_array2, index=stats_col2, columns=['Statistics'])

latextQ3_pre = r'''
**Q3**: When $g(X)=aX+b$, what can we say about
- $E(aX+b), E(X)$?
- $Var(aX+b)$ and $Var(X)$?
You may experiment with different values of $a$ and $b$ to vary the input/output of $g(X)=aX+b$ above, e.g. $a,b=0,1,2$.
'''
st.write(latextQ3_pre)

st.write(df_stats2.T.to_html(index=True), unsafe_allow_html=True)
latextQ3 = r'''
When $g(X)=aX+b$, where $a,b \in \mathbb{R}$, then from the above calculations, we can infer that
- $E(aX+b)=aE(X)+b$, 
- $Var(aX+b)=a^{2}Var(X)$, 
where it can be checked that 
- $E(aX)=aE(X), E(b)=b$
- $Var(aX)=a^{2}Var(X), Var(b)=0$.
'''

st.text_input("Enter your answer for Q3 here.")
if st.button('Click here to compare your answer against explanation of Q3'):
     st.write(latextQ3)
