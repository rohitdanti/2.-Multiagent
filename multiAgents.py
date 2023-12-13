# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newFood = successorGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #print(newGhostStates)
        #print(newFood.asList())
        #print('****************')
        Food_Locations=newFood.asList() #returns the coordinates of the food in the layout in a list format of (x,y)
        #print(Food)
        ghost_Position = successorGameState.getGhostPositions() # returns the ghost positions
        Distances_From_Ghost = [] #list to store the distances of ghost from the Pacman Agent
        Distances_From_Food = [] # list to store the distances of food from the Pacman agent

        for foodparticle in Food_Locations:
            manhattan_Dist_Food = abs(foodparticle[0] - newPos[0]) + abs(foodparticle[1] - newPos[1]) # calculating th manhattan distance between the current Pacman position and the food particles 
            Distances_From_Food = Distances_From_Food + [manhattan_Dist_Food] # storing all the calculated manhattan distances from the food particles in a list
        for ghost in ghost_Position:
            manhattan_Dist_Ghost = abs(ghost[0] - newPos[0]) + abs(ghost[1] - newPos[1]) # calculating the manhattan distance between the ghost and the pacman's current location
            Distances_From_Ghost = Distances_From_Ghost + [manhattan_Dist_Ghost] # storing all the calculated manhattan distances from the ghosts in a list
  
        plus_infinity = float("inf")
        minus_infinity = -float("inf")

        #if there's no food particles left, return plus infinity implying the game has been won
        if len(Distances_From_Food)==0:
            return plus_infinity
        
        #if the pacman stays in the same place then return minus infinity implying that pacman must keep moving
        if currentGameState.getPacmanPosition()==newPos:
            return minus_infinity
        
        #if pacman is very close to any of the ghosts, return minus infinity implying the game will be lost
        for dist in Distances_From_Ghost:
            if dist<1:
                return minus_infinity
            
        #return evaluation scores of the state by taking the sum of reciprocals of combined distances of each food particles from pacman and number of food particles 
        return ((1/sum(Distances_From_Food)) + (1/len(Distances_From_Food)))

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth):
            #as this is a max_Value functtion, we try to select the maximim score out of the available options.
            # w is set to minus infinity initially, later we check this value with all the scores we get from min_Value function and set the highest value to w. 
            # Act is also set to none initally, on finding the highest score we set the corresponding action to Act
            Act = None
            minus_infinity = -(float("inf")) 
            w = minus_infinity 
            pacman_agent_index = 0 #implying pacman
            ghost_agent_index = 1 #implying ghost
            Legal_Actions = gameState.getLegalActions(pacman_agent_index) #retrieving all possible actions for the pacman
            #print(Actions)
            
            #checking if current game state is a terminal states or if the defined depth for the depth limit search is reached
            if gameState.isLose() or gameState.isWin() or len(Legal_Actions) == 0 or depth == self.depth:         
                terminations_State_Utility = self.evaluationFunction(gameState) #get the state's utility
                return(terminations_State_Utility,None) # return the retrieved utility
                                                                                    
            #looping over all the legal actions
            for action in Legal_Actions:
                #calling the min_value from max_value function and vice versa making it a recursive call replicating the tree structure of the Minimax Tree
                next_State = gameState.generateSuccessor(pacman_agent_index,action) #calling pacman.py's generate successor func
                min_value_Result = min_value(next_State, ghost_agent_index, depth) #returns a list having score on 0th index and action on 1st index
                min_value_Score = min_value_Result[0] #fetching the score
                if(min_value_Score > w):
                    Act = action #setting the corresponding action of min value to Act
                    w = min_value_Score #setting the value of w to the value returned by the min_value func 
            return(w,Act)

        def min_value(gameState,agentID,depth):
            
            Legal_Actions = gameState.getLegalActions(agentID)
            plus_infinity = float("inf")
            num_Of_Legal_Actioins= len(Legal_Actions)
            num_Of_Agents = gameState.getNumAgents() #fetching number of agents from Pacman.py
            num_Of_Ghosts = num_Of_Agents -1 #total agents minus 1 pacman agent
            
            #initializing the action to null and l to plus infinity
            l = plus_infinity
            Act = None
             
            if num_Of_Legal_Actioins == 0: #checking if there are no legal actions
                terminations_State_Utility = self.evaluationFunction(gameState) #get the state's utility
                return(terminations_State_Utility,None) # return the retrieved utility
                                                                                             
            #looping over all legal actions
            for action in Legal_Actions:
                if(agentID == num_Of_Ghosts):
                    next_Ghost_Action = gameState.generateSuccessor(agentID,action)
                    max_Value_Result = max_value(next_Ghost_Action,depth + 1) #calls the max_Value by passing ghost's next action and depth at above level of the tree. This func returns a list having score on 0th index and action on 1st index
                else:
                    next_Pacman_Action = gameState.generateSuccessor(agentID, action) #calling pacman.py's generate successor func
                    max_Value_Result = min_value(next_Pacman_Action, agentID+1 , depth) #returns a list having score on 0th index and action on 1st index
                
                Score = max_Value_Result[0] #fetching the score
                
                #min_Value func selects the least score to reduce the opponent's chance of winning as this is a adversarial search.
                if(Score < l): 
                    l = Score #setting the value of w to the value returned by the min_value func
                    Act = action #setting the corresponding action of min value to Act 
            return(l,Act)
        return max_value(gameState,0)[1]
         

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        plus_infinity = float("inf")
        minus_infinity = -float("inf")
        pacman_agent_index = 0 #implying pacman
        ghost_agent_index = 1 #implying ghost
        
        #similar to minimax seach but the only difference being pruning to avoid unnecessary computations
        def max_value(gameState,depth,alpha,beta):
            
            Act = None
            w = minus_infinity
            Legal_Actions = gameState.getLegalActions(pacman_agent_index) # Get all the legal actions for the pacman agent from current state
            
            if gameState.isLose() or len(Legal_Actions)==0 or gameState.isWin() or depth==self.depth:
                terminations_State_Utility = self.evaluationFunction(gameState) #get the state's utility
                return (terminations_State_Utility, None) # return the retrieved utility

            #looping over all legal actions.
            for action in Legal_Actions: 
                Next_Pacman_Action = gameState.generateSuccessor(pacman_agent_index, action)
                min_Value_Result = min_value(Next_Pacman_Action, ghost_agent_index, depth, alpha, beta) #returns the list having score in 0th index and action on 1st index
                Score = min_Value_Result[0] #taking the evaluation score from min_Value result
                
                #in max_Value we select the action corresponding to highest evaluation score. 
                # As w is set to -infinity we compare it with each score and select the highest score and its action.
                if w < Score:
                    Act = action #setting the corresponding action of min value to Act
                    w = Score #setting the value of w to the value returned by the min_value func
                    
                if w > beta:
                    return (w, Act) # end the search and return when w is greater than beta
                if(w > alpha): #At max level, when the utility for a action is greater than alpha, we set alpha to that utility value 
                    alpha = w
                    
            return (w,Act)

        def min_value(gameState,agentIndex,depth,alpha,beta):
            
            Act = None
            l = plus_infinity
            Legal_Actions=gameState.getLegalActions(agentIndex) # Get all the legal actions of the ghost
            num_Of_Agents = gameState.getNumAgents()
            
            if len(Legal_Actions) == 0: #checking if there are no legal actions
                terminal_State_Utility = self.evaluationFunction(gameState) #get the state's utility
                return (terminal_State_Utility, None) #return the utility and none as the correspopnding action
            
            for action in Legal_Actions:
                next_Action = gameState.generateSuccessor(agentIndex,action)
                if (agentIndex == num_Of_Agents - 1): #check if agent is ghost
                    result = max_value(next_Action, depth + 1,alpha,beta) #calling max_Value when agent is ghost 
                else:
                    result = min_value(next_Action, agentIndex + 1, depth, alpha, beta) #calling min_Value when agent is Pacman
                Eval_score=result[0]
                
                if (Eval_score<l): #in min value, select the lowest utility and its action
                    Act = action
                    l = Eval_score
                    
                if (l<alpha): #when alpha is less than the utility value, return
                    return (l,Act)
                
                if(l<beta): #set the minimum value to beta
                    beta = l
            #print(l)
            #print(Act)        
            return(l,Act)

        alpha = minus_infinity
        beta = plus_infinity
        return max_value(gameState,0,alpha,beta)[1] # returns the 1st index which represents the action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth):
            pacman_index = 0
            ghost_index = 1
            minus_infinity = -(float("inf"))
            Act=None #initializing the action to null
            
            Actions = gameState.getLegalActions(pacman_index) #get all legal actions of the pacman agent
            if gameState.isWin() or gameState.isLose() or len(Actions)==0 or depth==self.depth:  #checking if the current state is a terminal state 
                Termination_eval_Score = self.evaluationFunction(gameState)
                return (Termination_eval_Score,None)                                  

            w = minus_infinity
            for action in Actions: #iterating over all the actions of the pacman agent
                Expectimax_Result=exp_value(gameState.generateSuccessor(0,action),ghost_index,depth) #calling the expectimax function
                Expectimax_Score=Expectimax_Result[0] #getting the score part of the result
                
                if(w<Expectimax_Score): #check if w is least else set the current action's score and action to w and Act
                    Act = action 
                    w = Expectimax_Score 
                #print(w)
                #print(Act)   
            return(w,Act)

        def exp_value(gameState,agentID,depth):
            Act=None #initially setting the actions to null
            l=0
            num_Of_Agents = gameState.getNumAgents()
            Actions = gameState.getLegalActions(agentID)
            
            if len(Actions)==0:
                Terminal_State_Score = self.evaluationFunction(gameState)
                return (Terminal_State_Score,None)
            
            for action in Actions:
                Agent_next_state = gameState.generateSuccessor(agentID,action)
                
                if(agentID == num_Of_Agents-1):
                    result = max_value(Agent_next_state , depth+1)
                else:
                    result = exp_value(Agent_next_state, agentID + 1, depth)
                
                Eval_Score = result[0]
                probability = Eval_Score/len(Actions)
                l = l + probability
                
                #print(Eval_Score)
                #print(probability)
                #print(l)
                
            return(l,Act)
        return max_value(gameState,0)[1]