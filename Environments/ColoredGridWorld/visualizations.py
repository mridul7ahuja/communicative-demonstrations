import itertools
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.path import Path
import matplotlib.patches as patches

def viewDictionaryStructure(d, dictionaryType, indent=0):
    if dictionaryType == "t":
        levels  = ["state", "action", "next state", "probability"]
    if dictionaryType == "r":
        levels  = ["state", "action", "next state", "reward"]
    if dictionaryType == "t_key":
        levels  = ["action", "next state", "probability"]
    if dictionaryType == "r_key":
        levels  = ["action", "next state", "reward"]

    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, dictionaryType, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))

"""
    Visualizes environment where the input is a set of states to visualize instead of a grid - can take in irregular state spaces.
    Inputs: 
        states: list of state tuples (x,y)
        goalStates: list of possible goals 
        trapStates: list of obstacle or trap spaces
        trajectory: list of states an agent travels through
        goalNameDictionary: dictionary where keys are state tuples and values are names of those states. Use if you want to name goals.
"""
def visualizeEnvironmentByState(states, goalStates = [], trapStates = [], trajectory = [], goalNameDictionary = {}, gridScale = 1.5):
    gridAdjust = .5

    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1.0))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0, end, 1.0))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #goal coloring and labeling
    for (goalx,goaly) in goalStates:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
        if (goalx, goaly) in goalNameDictionary.keys():
            ax.text(goalx-.15, goaly-.15, goalNameDictionary[(goalx, goaly)], fontsize = 12)
    
    for (x,y),colour in states.items():
        ax.add_patch(Rectangle((x-gridAdjust, y-gridAdjust), 1, 1, fill=True, color=colour, alpha=.1))

    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='red', alpha=.1))

    #trajectory path coloring
    for indx, (statex, statey) in enumerate(trajectory):
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=True, color='blue', alpha=.1))
        ax.text(statex-.1, statey-.1, str(indx), fontsize = 20)

    plt.show()



def visualizePolicy(states, policy, trueGoalState, otherGoals=[], trapStates=[], arrowScale = .3):
    #grid height/width
    gridAdjust = .5
    gridScale = 1.5
    
    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1.0))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0, end, 1.0))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #goal and trap coloring 
    ax.add_patch(Rectangle((trueGoalState[0]-gridAdjust, trueGoalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.5))

    for (goalx, goaly) in otherGoals:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='red', alpha=.1))

    #labeled values
    for (statex, statey), actionDict in policy.items():
        for (optimalActionX, optimalActionY), actionProb in actionDict.items():
            plt.arrow(statex, statey, optimalActionX*actionProb*arrowScale, optimalActionY*actionProb*arrowScale, head_width=0.05*actionProb, head_length=0.1*actionProb)    
    plt.show()

def visualizePolicyWithBarrier(states, policy, trueGoalState, barrierList, otherGoals=[], trapStates=[], arrowScale = .3):
    #grid height/width
    gridAdjust = .5
    gridScale = 1.5
    
    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #goal and trap coloring 
    ax.add_patch(Rectangle((trueGoalState[0]-gridAdjust, trueGoalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.5))
    for (goalx, goaly) in otherGoals:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='black', alpha=.1))
    for (statex, statey), (nextStatex, nextStatey) in barrierList:
        plt.arrow(statex, statey, (nextStatex-statex)*arrowScale, (nextStatey-statey)*arrowScale, head_width=0.05, head_length=0.1, color = 'red')
    #labeled values
    for (statex, statey), actionDict in policy.items():
        for (optimalActionX, optimalActionY), actionProb in actionDict.items():
            plt.arrow(statex, statey, optimalActionX*actionProb*arrowScale, optimalActionY*actionProb*arrowScale, head_width=0.05*actionProb, head_length=0.1*actionProb)    
    plt.show()


"""
    Visualizes policy of a given belief state where the input is a set of states to visualize instead of a grid - can take in irregular state spaces.
    Inputs: 
        states: list of state position tuples (x,y)
        policy: probability of an action given the state s is of form (positionx, positiony), belief
        belief: the belief state for which to visualize the policy
        goalStates: list of possible goals 
        trapStates: list of obstacle or trap spaces
        trajectory: list of states an agent travels through
"""
def visualizePolicyOfBeliefByState(states, policy, belief, goalStates = [], trapStates = [], trajectory = [], arrowScale = .3):
    gridAdjust = .5
    gridScale = 1.5
    
    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #goal and trap coloring 
    for (goalx,goaly) in goalStates:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='red', alpha=.1))

    #trajectory path coloring
    for indx, (statex, statey) in enumerate(trajectory):
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=True, color='blue', alpha=.1))
        ax.text(statex-.1, statey-.1, str(indx), fontsize = 25)

    #labeled values
    for ((statex, statey), b) in policy.keys():
        if b == belief:
            for (actionx, actiony), actionProb in policy[((statex, statey), b)].items():
                plt.arrow(statex, statey, actionx*actionProb*arrowScale, actiony*actionProb*arrowScale, head_width=0.05*actionProb, head_length=0.1*actionProb)    
    plt.show()

def visualizeValueTable(gridWidth, gridHeight, goalState, trapStates, valueTable):
    gridAdjust = .5
    gridScale = 1.5
    
    xs = np.linspace(-gridAdjust, gridWidth-gridAdjust, gridWidth+1)
    ys = np.linspace(-gridAdjust, gridHeight-gridAdjust, gridHeight+1)
    
    plt.rcParams["figure.figsize"] = [gridWidth*gridScale,gridHeight*gridScale]
    ax = plt.gca(frameon=False, xticks = range(gridWidth), yticks = range(gridHeight))

    #goal and trap coloring 
    ax.add_patch(Rectangle((goalState[0]-gridAdjust, goalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='red', alpha=.1))
    
    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color = "black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color = "black")

    #labeled values
    for (statex, statey), val in valueTable.items():
        plt.text(statex-.2, statey, str(round(val, 3)))    

    plt.show()

def visualizeValueTableMultipleGoals(gridWidth, gridHeight, goalState, otherGoals, trapStates, valueTable):
    gridAdjust = .5
    gridScale = 1.5
    
    xs = np.linspace(-gridAdjust, gridWidth-gridAdjust, gridWidth+1)
    ys = np.linspace(-gridAdjust, gridHeight-gridAdjust, gridHeight+1)
    
    plt.rcParams["figure.figsize"] = [gridWidth*gridScale,gridHeight*gridScale]
    ax = plt.gca(frameon=False, xticks = range(gridWidth), yticks = range(gridHeight))

    #goal and trap coloring 
    ax.add_patch(Rectangle((goalState[0]-gridAdjust, goalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.5))

    for (goalx, goaly) in otherGoals:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))

    
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='black', alpha=.1))
    
    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color = "black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color = "black")

    #labeled values
    for (statex, statey), val in valueTable.items():
        plt.text(statex-.2, statey, str(round(val, 3)))    

    plt.show()


def visualizeTransitionTable(states, transitionTable, actionOfInterest, arrowScale = .5):

    colorDict = {(1,0): 'b', (0,1): 'g', (-1,0): 'r', (0,-1): 'm', (0,0): 'c'}

    #grid height/width
    gridAdjust = .5
    gridScale = 1.5
    
    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #labeled values
    for (statex, statey), actionDict in transitionTable.items():
        for (nextStatex, nextStatey), actionProb in actionDict[actionOfInterest].items():
            arrowColor = colorDict[(nextStatex-statex, nextStatey-statey)]
            plt.arrow(statex, statey, (nextStatex-statex)*actionProb*arrowScale, (nextStatey-statey)*actionProb*arrowScale, 
                head_width=0.05*actionProb, 
                head_length=0.1*actionProb, 
                color = arrowColor)    
    plt.show()
            
"""
    Visualizes the environment and all the different trajectories given as input
    Inputs:
        states: list of state position tuples (x,y)
        goalStates: list of possible goals 
        trapStates: list of obstacle or trap spaces
        trajectories: list of all trajectories to be plotted
        goalNameDictionary: dictionary where keys are state tuples and values are names of those states. Use if you want to name goals.     
        gridScale: to change the scale of the environment
        jitterMean: mean of the normal distribution from which the slight shift in the plotted trajectories are sampled from 
        jittterVar: variance of the normal distribution from which the slight shift in the plotted trajectories are sampled from 
"""
            
def visualizeEnvironmentMultipleTraj(states, goalStates = [], trapStates = [], trajectories = [], goalNameDictionary = {}, gridScale = 1.5, jitterMean = 0, jitterVar = 0.03):
    gridAdjust = .5

    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]
    
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon=False, xticks = range(minimumx-1, maximumx+2), yticks = range(minimumy-1, maximumy+2))
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1.0))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0, end, 1.0))

    #gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    #goal coloring and labeling
    for (goalx,goaly) in goalStates:
        ax.add_patch(Rectangle((goalx-gridAdjust, goaly-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
        if (goalx, goaly) in goalNameDictionary.keys():
            ax.text(goalx-.15, goaly-.15, goalNameDictionary[(goalx, goaly)], fontsize = 12)
    
    for (x,y),colour in states.items():
        ax.add_patch(Rectangle((x-gridAdjust, y-gridAdjust), 1, 1, fill=True, color=colour, alpha=.1))

    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='red', alpha=.1))

    #trajectory path drawing for multiple trajectories
    for trajectory in trajectories:
        jitter = np.random.normal(jitterMean, jitterVar, 2)
        verts = []
        for indx, (statex, statey) in enumerate(trajectory):
            if( indx!=0 and indx!=(len(trajectory)-1) ):
                verts.append((statex + jitter[0], statey + jitter[1]))
            else:
                verts.append((statex, statey))
        codes = [Path.MOVETO]
        for i in range(len(verts)-1):
            codes.append(Path.LINETO)
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='None', lw=0.5, capstyle = 'butt')
        for indx, (statex, statey) in enumerate(trajectory):
            ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, alpha=.1))
            ax.add_patch(patch)

    plt.show()
    
"""
    Graph for probability at each time step for different objects in the given dictionary
    Input:
        probAtEachStepDict: dictionary of _: list of probability at each time step
        title, xlabel: Title for X-axis, ylabel: Title for Y-axis
"""
    
def plotProbAtEachStep(probAtEachStepDict, title="", xlabel = "Time Point", ylabel = "Posterior Probability"):
    X = [i for i in range(1,len(list(probAtEachStepDict.values())[0])+1)]
    for env in probAtEachStepDict.keys():
        Y = probAtEachStepDict[env]
        plt.plot(X,Y, label = env, linestyle='dashed',marker='o', markersize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
"""
    Visualizes the belief states of the trajectory by plotting the pmf over the possible colour rewards at each step in the form a bar chart
    Inputs: 
        dictAtEachStep: list of beliefStates(dictionaries) over the trajectory at each step 
        figSize: tuple for the figure size of the entire plot
        showHeight: boolean for showing the heights of each bar or not
"""
   
def plotDictAtEachStep(dictAtEachStep, figSize = (10,20), majorYTick = 0.5, minorYTick = 0.25, showHeight = False):
    rows = len(dictAtEachStep)
    fig, ax = plt.subplots(rows, 1, sharex = 'col', sharey = 'col', figsize= figSize)
    for i in range(len(dictAtEachStep)):
        D = dictAtEachStep[len(dictAtEachStep)-1-i]
        lists = sorted(D.items()) 
        x, y = zip(*lists) 
        ax[i].bar(range(len(D)), list(y), align='center', tick_label = list(x))
        ax[i].yaxis.set_major_locator(MultipleLocator(majorYTick))
        ax[i].yaxis.set_minor_locator(MultipleLocator(minorYTick))
        if(showHeight):
            for p in ax[i].patches:
                ax[i].annotate(str(round(p.get_height(), 3)), (p.get_x() * 1.005, p.get_height() * 1.05))
        if(i==(len(dictAtEachStep) -1)):
            ax[i].set(xlabel='Reward Function', ylabel="Step " + str(len(dictAtEachStep) - i-1))
        else:
            ax[i].set(ylabel="Step " + str(len(dictAtEachStep) - i-1)) 
        