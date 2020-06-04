import multiprocessing
import itertools
import time

class ActionTransition():
    def __init__(self, actionSpace, transitionFunction):
        self.actions = actionSpace
        self.getTransition = transitionFunction

    def __call__(self, jointState):
        return({action: self.getTransition(action, jointState) for action in self.actions})

def getNextState(action, currentState): 
    return(tuple([s**2 for s in currentState]))


def main():
    # Toy setup
    gridWidth = 5000
    gridHeight = 1000
    toyStateSpace = set(itertools.product(range(gridWidth), range(gridHeight)))
    actions = [(-1,0), (1,0), (0,1), (0,-1)]
    getActionDict = ActionTransition(actions, getNextState)
    goalState = (0,0)

    print(multiprocessing.cpu_count()) #12 on my device

    """
    From my understanding this parallelizes the outer loop - i.e. many action dictionaries are created in parallel. This required creating a new function/class
    to represent that dictionary generation as a function (ActionTransition class instantiated with getActionDict function). 

    """
    # With parallelization
    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as executor:
        transitionTable = {state: actionDict for state, actionDict in zip(toyStateSpace, executor.map(getActionDict, toyStateSpace)) if state != goalState }
    print("--- %s seconds ---" % (time.time() - start_time))

    #without parallelization
    start_time = time.time()
    transitionTable_NonParallel = {state: getActionDict(state) for state in toyStateSpace if state != goalState }
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
	main()