import numpy as np
import matplotlib.pyplot as plt
from numpy import *

class equation():
    def __init__(self, eq_str):
        
        #Get left and right hand side
        self.eq_str = "".join(eq_str.split(" "))
        self.eq_lhs, self.eq_rhs = self.eq_str.split("=")
        
        #Check if we have a first-order differential equation
        #If so, transform to different equation for simulation
        if(self.eq_lhs[0])=="D":
                self.eq_lhs = self.eq_lhs[1:]
                self.eq_rhs = "%s + dt*(%s)"%(self.eq_lhs,self.eq_rhs)
        
        #Check if we have explicit time indexing in lhs and strip out
        if(len(self.eq_lhs)>3):
            if(self.eq_lhs[-3:]=="[t]" or self.eq_lhs[-3:]=="[0]"):
                self.eq_lhs = self.eq_lhs[:-3]
        
        #Store the equation
        self.eq_dict = {"lhs":self.eq_lhs, "rhs":self.eq_rhs}
        
    def get_dict(self):
        return(self.eq_dict)
    
    def get_str(self):
        return(self.eq_str)
    
    def get_rhs(self):
        return(self.eq_rhs)

    def get_lhs(self):
        return(self.eq_lhs)
    
    def __str__(self):
        return(self.eq_str)
        
class equation_system():
    def __init__(self, eq_strs):
        self.system_strs = "".join(eq_strs.split(" "))
        if(";" in eq_strs):        
            self.system_list = self.system_strs.split(";")
        else:
            self.system_list = self.system_strs.split(",")
            
        self.system_dict={}
        self.system_vars = []
        for eq_str in self.system_list:
            eq  = equation(eq_str)
            lhs = eq.get_lhs()
            self.system_dict[lhs] = eq
            self.system_vars.append(lhs)
            
    def get_vars(self):
        return(self.system_vars)
    
    def get_eq(self, var):
        return(self.system_dict[var].get_rhs())
    
    def __str__(self):
        l=[]
        for name,eq in self.system_dict.items():
            l.append(eq.get_str())
        s="\n".join(l)
        return(s)
        

class simulator():
    def __init__(self, init_system, dynamics_system):

        #Make init and dynamics systems
        self.init_system     = equation_system(init_system)
        self.dynamics_system = equation_system(dynamics_system)
        self.vars            = self.dynamics_system.get_vars()
        
        #Initialize the system state,
        self.state = {}
        for var in self.vars:
            self.state[var] = eval(self.init_system.get_eq(var))
    
    def __str__(self):
        l=[]
        s=self.init_system.__str__() + "\n"
        s=s+self.dynamics_system.__str__()
        return(s)

    def simulate(self,start=0, stop=10, delta=1, debug=True):
        
        state_history = state={"ts":np.arange(start,stop,delta)}
        steps = len(state_history["ts"])
                
        for var in self.vars:
            state_history[var]         = np.zeros(steps)
        
        for i,t in enumerate(state_history["ts"]):
                        
            state_history["i"]  = i
            state_history["t"]  = t
            state_history["dt"] = delta 

            for var in self.vars:
                
                state_history["n_eps"]    = np.random.randn()
                state_history["u_eps"]    = np.random.rand()
                
                if(i==0):
                    eq = self.init_system.get_eq(var)
                else:
                    eq = self.dynamics_system.get_eq(var)
                
                state_history[var][i] = eval(eq, globals(), state_history)
            
            state = {var:state_history[var][i] for var in self.vars}
            if(debug): print("Step:%d  t:%f  State:"%(i,t), state)
                
        self.state_history=state_history
        
    def plot(self,v=None, ylim=None):
        
        plt.figure(figsize=(16,4))
        
        if(v is None):
            v=self.vars
        else:
            v = v.split(",")
            
        for var in v:
            if(var[0]=="D"):
                l = "d%s/dt"%(var[1:])
            else:
                l = var
                
            #Check for binary variable
            vals = np.unique(self.state_history[var])
            if(len(vals)==2 and (0 in vals) and (1 in vals)):
                #plot as points
                ind = self.state_history[var]==1
                plt.plot(self.state_history["ts"][ind],0*self.state_history[var][ind],'o',label=l)                
            else:
                #Plot as a line
                plt.plot(self.state_history["ts"],self.state_history[var],label=l)

        if(ylim is not None):
            plt.ylim(ylim)
        
        plt.legend()
        plt.grid(True)
        plt.xlabel("Time (t)")
        plt.title("Simulation Output")
        plt.show()