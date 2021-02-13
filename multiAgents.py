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
        # print(legalMoves)
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(currentGameState)
        # print(successorGameState)
        # print(newPos)
        # print(newFood.asList())
        # print(newGhostStates)
        # print(successorGameState.getScore())
        # print(newScaredTimes)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        from util import manhattanDistance

        food = newFood.asList()
        foodDistances = []
        ghostDistances = []
        count = 0.0 

        for item in food:
            foodDistances.append(manhattanDistance(newPos,item))

        for i in foodDistances:
            if i <= 4:
                count += 1/(float(i))
            elif i > 4 and i <= 15:
                count +=  1/(float(i))
            else:
                count +=  1/(float(i))

        for ghost in successorGameState.getGhostPositions():
            ghostDistances.append(manhattanDistance(ghost,newPos))

        #  100 and 20 did well... 1000 and 20... 1000 and 100
        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos:
                count = count - 1000

            elif manhattanDistance(ghost,newPos) <= 3.5:
                count = count - 20.0/float(manhattanDistance(ghost,newPos))
        # return count
        return successorGameState.getScore() + count

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

        # print(self.depth)
        # print(gameState.isWin())
        # print(gameState.isLose())
        # print(gameState.getNumAgents())
        # print(gameState.generateSuccessor(0,"Stop"))
        # print(gameState.getLegalActions(0))
        # print(self.evaluationFunction(gameState))
        
        def miniMaxAlgo(agentNum,currDepth,gameState):
            numAgents = gameState.getNumAgents()
            agentActions = gameState.getLegalActions(agentNum)
            cost = None
            action =None
            flag = 0
            # depth reahced maximum
            if(currDepth == self.depth):
                return (self.evaluationFunction(gameState),0)
            # no more actions left
            if(len(agentActions)==0):
                return (self.evaluationFunction(gameState),0)
            
            # finding the next agent and the depth if it changes.
            succesorANum = (agentNum + 1)%numAgents
            if(agentNum == numAgents-1):
                currDepth = 1 + currDepth
                

            for currentAc in agentActions:
                if(flag ==1):
                    currNodeValue = cost
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = miniMaxAlgo(succesorANum,currDepth,newGameState)
                    newCost = newReturn[0]
                    if agentNum != 0:
                        if(newCost<currNodeValue):
                            cost = newCost
                            action = currentAc
                    elif agentNum ==0:
                        if(newCost>currNodeValue):
                            cost = newCost
                            action = currentAc
                elif(flag == 0 ):
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = miniMaxAlgo(succesorANum,currDepth,newGameState)
                    cost = newReturn[0]
                    action = currentAc
                    flag =1
            return (cost,action)
        ActionToBeReturned = miniMaxAlgo(0,0,gameState)[1]
        return ActionToBeReturned

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # print(self.depth)
        # print(gameState.isWin())
        # print(gameState.isLose())
        # print(gameState.getNumAgents())
        # print(gameState.generateSuccessor(0,"Stop"))
        # print(gameState.getLegalActions(0))
        # print(self.evaluationFunction(gameState))
        
        def alphaBetaPruningAlgo(alpha,beta,agentNum,currDepth,gameState):
            numAgents = gameState.getNumAgents()
            agentActions = gameState.getLegalActions(agentNum)
            cost = None
            action =None
            flag = 0
            # depth reahced maximum
            if(currDepth == self.depth):
                return (self.evaluationFunction(gameState),0)
            # no more actions left
            if(len(agentActions)==0):
                return (self.evaluationFunction(gameState),0)
            
            # finding the next agent and the depth if it changes.
            succesorANum = (agentNum + 1)%numAgents
            if(agentNum == numAgents-1):
                currDepth = 1 + currDepth
                

            for currentAc in agentActions:
                if(flag ==1):
                    # prune if possible
                    currNodeValue = cost
                    if(agentNum!=0):
                        if(currNodeValue<alpha):
                            return (cost,action)
                    elif(agentNum==0):
                        if(currNodeValue>beta):
                            return (cost,action)
                    # else do as before
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = alphaBetaPruningAlgo(alpha,beta,succesorANum,currDepth,newGameState)
                    newCost = newReturn[0]
                    if agentNum != 0:
                        if(newCost<currNodeValue):
                            cost = newCost
                            action = currentAc
                            # b changes as per cost
                            if(cost<beta):
                                beta = cost
                    elif agentNum ==0:
                        if(newCost>currNodeValue):
                            cost = newCost
                            action = currentAc
                            if(cost>alpha):
                                alpha = cost

                elif(flag == 0 ):
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = alphaBetaPruningAlgo(alpha,beta,succesorANum,currDepth,newGameState)
                    cost = newReturn[0]
                    action = currentAc
                    flag =1
                    if(agentNum != 0):
                        if(cost<beta):
                            beta = cost
                    elif(agentNum == 0):
                        if(cost>alpha):
                            alpha = cost


            return (cost,action)
        ActionToBeReturned = alphaBetaPruningAlgo(-float("inf"),float("inf"),0,0,gameState)[1]
        return ActionToBeReturned

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
        # print(self.depth)
        # print(gameState.isWin())
        # print(gameState.isLose())
        # print(gameState.getNumAgents())
        # print(gameState.generateSuccessor(0,"Stop"))
        # print(gameState.getLegalActions(0))
        # print(self.evaluationFunction(gameState))
        
        def expectiMaxAlgo(agentNum,currDepth,gameState):
            numAgents = gameState.getNumAgents()
            agentActions = gameState.getLegalActions(agentNum)
            cost = None
            action =None
            flag = 0
            # depth reahced maximum
            if(currDepth == self.depth):
                return (self.evaluationFunction(gameState),0)
            # no more actions left
            if(len(agentActions)==0):
                return (self.evaluationFunction(gameState),0)
            
            # finding the next agent and the depth if it changes.
            succesorANum = (agentNum + 1)%numAgents
            if(agentNum == numAgents-1):
                currDepth = 1 + currDepth
                

            for currentAc in agentActions:
                if(flag ==1):
                    currNodeValue = cost
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = expectiMaxAlgo(succesorANum,currDepth,newGameState)
                    newCost = newReturn[0]
                    # all the actions have equal probabiility and that is used to determine cost
                    if agentNum != 0:
                        p = len(agentActions)
                        cost = cost + float(newCost)/float(p)
                        action = currentAc
                    elif agentNum ==0:
                        if(newCost>currNodeValue):
                            cost = newCost
                            action = currentAc
                elif(flag == 0 ):
                    newGameState  = gameState.generateSuccessor(agentNum,currentAc)
                    newReturn = expectiMaxAlgo(succesorANum,currDepth,newGameState)
                    cost = newReturn[0]
                    action = currentAc
                    # all the actions have equal probabiility and that is used to determine cost
                    if(agentNum!=0):
                        p = len(agentActions)
                        cost= float(cost)/float(p)
                    flag =1
            return (cost,action)
        ActionToBeReturned = expectiMaxAlgo(0,0,gameState)[1]
        return ActionToBeReturned

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 

    > Better function has been found by tuning the function in partI further.
    > Various features have been used to find a good evaluation function.
    > Each feature has its own associated weight which is choosen in such a manner that the score improves.
    > The features used are:
        1. Location of pacman
        2. Location of food
        3. Location of capsules
        4. Location of active ghosts
        5. Location of scared ghosts
    > Capsules have been given more weight than food because the pacmans score will increase 
      more when it eats a capsule than food.
    > This type of reasoning is used to determine which feature will get what weight
    > Score can be increased by eating a scared ghost too.
    > The weight of capsules and weight of scared ghost is almost same as they 
      yeild similar increase in scores.

    """
    "*** YOUR CODE HERE ***"


    
    PacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    finalScore = 0
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # for item in foodList:
    #     finalScore = finalScore + 2/manhattanDistance(item,PacmanPosition)
    # for capsule in capsules:
    #     finalScore = finalScore + 100/manhattanDistance(capsule,PacmanPosition)
    # return currentGameState.getScore() + finalScore
    # print(currentGameState)
    # print(successorGameState)
    # print(newPos)
    # print(newFood.asList())
    # print(newGhostStates)
    # print(successorGameState.getScore())
    # print(newScaredTimes)
    foodValue = 0
    foods = currentGameState.getFood().asList()
    for food in foods:
        foodValue = max(foodValue,2/manhattanDistance(food,PacmanPosition))
    finalScore = finalScore + foodValue

    capusuleValue = 0
    capsules = currentGameState.getCapsules()
    for capsule in capsules:
        capusuleValue = max(capusuleValue,30/manhattanDistance(capsule,PacmanPosition))
    finalScore = finalScore + capusuleValue
    finalScore = finalScore + sum(scaredTimes)
    activeGhosts = 0
    scaredGhosts = 0
    # activeGhostsDistances =[]
    # for ghost in ghostStates:
    #     if ghost.scaredTimer:
    #         scaredGhosts+=1
    #     else:
    #         activeGhosts +=1
    # finalScore = finalScore - 8/(activeGhosts+1)
    # activeGhostValue = 0
    # for item in activeGhosts:
    #     activeGhostsDistances.append(manhattanDistance(PacmanPosition,item.getPosition()))
    # finalScore = finalScore - activeGhostValue
    # scaredGhostValue = 0
    # for item in scaredGhosts:
    #     scaredGhostValue = max(scaredGhostValue,50/manhattanDistance(item.getPosition(),PacmanPosition))
    # finalScore = finalScore +scaredGhostValue
    # cur =0 
    # for item in activeGhostsDistances:
    #     if item < 3 :
    #         finalScore = finalScore - 1
    
    return currentGameState.getScore() + finalScore

# Abbreviation
better = betterEvaluationFunction
