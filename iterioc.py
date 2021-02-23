# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 02:31:16 2021

@author: Hamadeh
"""
import np
import copy
def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_
def argmax(L):
    return int(np.random.choice(allmax(L)))
def comp(x,y):
    return (x-y)%3
class iterioc():
    def __init__(self,subagent,n):
        self.i=subagent()
        self.u=subagent()
        self.n=n
        self.L=np.zeros((2*n,6))
        #each agent reports sucess predicting self and others here the next takes
        #first is A0 and E0 predicting him,
        #then A0 and E0 predicting me
        self.perlast=np.zeros(2*n+2,dtype=int)
        #laset A0, E0
        #last choice shifted by 1 including agents excl last since no judge
        self.perlast-=1
        self.ilast=-1
        self.ulast=-1
        self.temp=[]
    def ag(self,obs,conf):
        obsu=copy.deepcopy(obs)
        if(obs.step!=0):
            self.temp.append(obs.lastOpponentAction)
            self.ulast=obs.lastOpponentAction #other at end of code
            self.u.tweeze(self.ulast)
            self.i.tweeze(self.ilast)
            obsu.lastOpponentAction=self.ilast #other implicit
            for i in range(self.n):
                idi=2*i #idi is predicting enemy index
                idu=idi+1
                #first two how me and adversarial predict u
                self.L[idi][comp(self.ulast,self.perlast[idi])]+=1
                self.L[idi][3+comp(self.ulast,self.perlast[idu])]+=1
                #second two how adverserial and me predic me
                self.L[idu][3+comp(self.ilast,self.perlast[idi])]+=1
                self.L[idu][comp(self.ilast,self.perlast[idu])]+=1 
                          
        ansi=self.i.ag(obs,conf)
        ansu=self.u.ag(obsu,conf)
        self.perlast[0]=ansi
        self.perlast[1]=ansu
        for i in range(self.n):
            #A1 looks at what was reported in terms of move and reputaiton
            #same for E1
            idi=2*i
            idu=idi+1
            c=argmax(self.L[idi]) #perhaps all max, must manage random beginnings 
            #c=int(np.random.choice([0,1,2,3,4,5],p=(self.L[idi]+1)/(sum(self.L[idi])+6),size=1))
            if(c<3):
                ichoice=(self.perlast[idi]+c+1)%3
            else:
                ichoice=(self.perlast[idu]+c+1)%3
            self.perlast[idi+2]=ichoice
            c=argmax(self.L[idu])  
            #c=int(np.random.choice([0,1,2,3,4,5],p=(self.L[idu]+1)/(sum(self.L[idu])+6),size=1))
            if(c<3):
                uchoice=(self.perlast[idu]+c+1)%3
            else:
                uchoice=(self.perlast[idi]+c+1)%3
            self.perlast[idu+2]=uchoice
        self.ilast=self.perlast[2*self.n]
        self.perlast = self.perlast.astype(int)
        return ((self.perlast[::2]-1)%3)
        #if(obs.step==998):
            #print(self.L)
            #print(self.perlast)
        #return int(int(self.ilast-1)%3)
    def tweeze(self,voted):
        self.ilast=voted