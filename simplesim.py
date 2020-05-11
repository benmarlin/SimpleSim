import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import ipywidgets as widgets
from IPython.display import clear_output
import json
import os, glob
from IPython.display import display, HTML
from IPython.display import display, Javascript
import time
import asyncio

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

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
        elif("\n" in eq_strs):
            self.system_list = self.system_strs.split("\n")
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
        
    def get_strs(self):
        return(self.system_strs)
    
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
        
    def plot(self,plot_vars=None, ylim=None, ax=None, space=None, title=None):
        
        
        if(plot_vars is None):
            vs=[self.vars]
        else:
            plot_vars = "".join(plot_vars.split(" "))
            vs        = []
            vstrs     = plot_vars.split("\n")
            for vstr in vstrs:
                vs.append(vstr.split(","))
        
        if(title is not None):
            titles = title.split("\n")
        else:
            titles = ["Simulation Output"]
                
        for i,var_set in enumerate(vs):
            
            
            plt.figure(figsize=(10,4))
                
            for var in var_set:
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
        
            if(len(titles)>1):                
                plt.title(titles[i])
            else:
                plt.title(titles[0])
        
            if(space is not None):
                with space:
                    plt.show()
            else:
                plt.show()
        
class gui():
    
    def __init__(self):
    
        
        if(IN_COLAB):
            pass
        else:
            if(not os.path.isdir("models")):
                os.mkdir("models")

        self.response = None                
        
        self.model_list = self.list_models()
    
        self.box_layout = widgets.Layout(display='flex',
                        flex_flow='column')
            
        self.wo = widgets.Output()
    
        self.wl = widgets.Label(value="Simulator Menu")

        self.wml = widgets.Dropdown(
            options= self.model_list,
            value="AutoregressiveDecay.json",
            description='Load:',
            disabled=False,
        )

        self.wn = widgets.Text(
            value='',
            description='Name:',
            disabled=False
        )
    
        self.wi = widgets.Textarea(
            value='',
            description='Initial State:',
            disabled=False
        )

        self.wd = widgets.Textarea(
            value='',
            description='Dynamics:',
            disabled=False
        )

        self.wsteps = widgets.IntText(
            value=100,
            description='Steps:',
            disabled=False
        )

        self.wv = widgets.Textarea(
            value='',
            description='Plot Vars:',
            disabled=False
        )

        self.wt = widgets.Textarea(
            description='Plot Titles:',
            value='',
            disabled=False
        )

        self.wbsim = widgets.Button(
            description='Start',
            disabled=False,
            button_style='', 
            tooltip='Click me to start simulation',
            #layout=widgets.Layout(width='100%')  
        )

        self.wbclear = widgets.Button(
            description='Clear',
            disabled=False,
            button_style='', 
            tooltip='Click me to start simulation',
            layout=widgets.Layout(width='75%') 
        )
        
        self.wbsave = widgets.Button(
            description='Save',
            disabled=False,
            button_style='', 
            tooltip='Click me to save model',
            layout=widgets.Layout(width='75%') 
        )
        
        self.wstatus = widgets.Label(
            value='Status:',
            description='Status',
            disabled=False
        )
        
        self.wbsim.on_click(self.start_sim)
        self.wbclear.on_click(self.clear_sim)
        self.wbsave.on_click(self.start_save)
        self.wml.observe(self.select_model)
        
        self.load("AutoregressiveDecay.json")

        self.display_sim()
    
    def set_status(self, status):
        self.wstatus.value="Status: %s"%status
    
    def refresh_model_list(self):
        self.model_list = self.list_models()
        self.wml.options = self.model_list
    
    def select_model(self,change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.load(self.wml.value)  
        
    def start_sim(self,b):
        s=simulator(self.wi.value, self.wd.value)
        s.simulate(stop=self.wsteps.value, debug=False)
        self.clear_sim(None)
        s.plot(space=self.wo, plot_vars =self.wv.value,  title=self.wt.value)

    def clear_sim(self,b):
        self.wo.clear_output()
        #self.display_sim()

    def display_sim(self):
        self.mbox = widgets.HBox([self.wbsim, self.wbclear, self.wbsave])
        self.cbox = widgets.VBox([self.wl, self.wml, self.wn, self.wi, self.wd, self.wsteps,self.wt, self.wv,self.mbox, self.wstatus],layout=self.box_layout)
        self.box  = widgets.HBox([self.cbox, self.wo])
        display(self.box)
        
    def start_save(self,b):
        #Check if file exists and don't overrite
        #Unless OK is checked
        save_name = "".join(self.wn.value.split(" ")) + ".json"
        save_path = 'models/%s'%(save_name)
                
        if(os.path.isfile(save_path)):
            self.get_response("finish_save()")
        else:
            self.finish_save()        
        
    def finish_save(self):
        save_dict={}
        save_dict["name"]=self.wn.value
        save_dict["init"]=self.wi.value
        save_dict["dynamics"]=self.wd.value
        save_dict["titles"]=self.wt.value
        save_dict["steps"]=self.wsteps.value
        save_dict["plot_vars"]=self.wv.value
        save_name = "".join(save_dict["name"].split(" ")) + ".json"
        save_path = 'models/%s'%(save_name)
        
        with open(save_path, 'w') as outfile:
            json.dump(save_dict, outfile)
        self.refresh_model_list()
        self.wml.value = save_name
        self.set_status("Model %s saved"%save_name)


    def get_response(self, callback):
        
        js = """var retVal = confirm('This model file already exists. Do you want to replace it?')
                var kernel = IPython.notebook.kernel;
                if(retVal==true){
                    var pyCommand = "global thisgui; g.%s";
                    kernel.execute(pyCommand);
                }
            """%callback
                
        global thisgui
        thisgui = self
        display(Javascript(js))
        
    def load(self, file_name):
        
        self.wo.clear_output()
        
        with open('models/%s'%file_name, 'r') as infile:
            save_dict=json.load(infile)

        self.wn.value=save_dict["name"]
        self.wi.value=save_dict["init"]
        self.wd.value=save_dict["dynamics"]
        self.wt.value=save_dict["titles"]
        self.wv.value=save_dict["plot_vars"]
        self.wsteps.value = save_dict["steps"]
        
        self.start_sim(None)
        
        self.set_status("Loaded model %s"%file_name)
        
    def list_models(self):
        l=[]
        for file in os.listdir("models"):
            if file.endswith(".json"): 
                l.append(file)
        l.sort()
        return(l)   
         
        