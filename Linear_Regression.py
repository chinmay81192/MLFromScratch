# -*- coding: utf-8 -*-
"""
\\ /// /

Created on Tue Jul 31 19:01:04 2018

@author: Chinmay
"""



x=[1,2,3,4,5]
y=[5,10,15,20,25]

t0=0.5
t1=0.7
alpha=0.5

while(True):
    
    sqr_diff = 0
    norm_diff = 0
    norm_diff_t0 = 0
    for x1,y1 in zip(x,y):
        result=t0+t1*x1
        norm_diff = norm_diff + (result - y1)*x1
        #sqr_diff = sqr_diff + (result - y)**2 
        norm_diff_t0 = norm_diff_t0 + (result - y1)
    #cost = 1/(2*len(y))*sqr_diff

    par_diff = 1/len(y)*norm_diff
    par_diff1 = 1/len(y)*norm_diff_t0
    if(abs(par_diff) < 0.001 and abs(par_diff1) < 0.001):
        break
    t0 = t0 - alpha * par_diff1
    t1 = t1 - alpha * par_diff

y1 = []
for x1 in x:
    y2 = t0 + t1 * x1
    y1.append(y2)
print(y1)


#Testing
a = 12
ytest = t0 + t1 * a
print(ytest)

    
