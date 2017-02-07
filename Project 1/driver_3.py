# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:23:23 2017

@author: Han-chung Lee
"""
"""
Begin by writing a class to represent the state of the game at a given turn, 
including parent and child nodes. We suggest writing a separate solver class to
work with the state class. Feel free to experiment with your design, for 
example including a board class to represent the low-level physical 
configuration of the tiles, delegating the high-level functionality to the 
state class.
"""
import argparse
#from hashlib import sha1
import numpy as np
#import queue as queue

    
class Queue:
    def __init__(self):
        self.A = []
        self.start = 0
    
    def empty(self):
        return not bool(len(self.A) - self.start)#self.__len__
    
    def extend(self, items):
        self.A.extend(items)              
           
    def enqueue(self, item):
        #print('hai enqueue')
        self.A.append(item)
    
    def dequeue(self):
        print('queue start', self.start)
        print('queue len',len(self.A))
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e
    
    def __eq__(self, item):
        return set(item.A) == set(self.A)
    
    def __contains__(self, item):
        if self.A is None:
            raise TypeError('not indexable')
        #return (item in )
        return item in self.A[self.start:]
    
    def __len__(self):
        print('check queue length', len(self.A) - self.start)
        return len(self.A) - self.start
    
class NPuzzle(object):
    
    """ Low-level physical configuration of N-Puzzle tiles."""
    
    def __init__(self, board):
        self.board = board
        
    def actions(self, state):
        """ Actions available in the current puzzle. Find the position of 
        zero first, and check if zero can be moved in up, down, left, right 
        directions. N-puzzle has edge and does not wrap around board. Returns 
        a list of available actions."""
        x, y = np.where(state.board == 0)
        action_list = []
        if (x > 0):
            action_list.append('UP')
        if (x < state.board.shape[0] - 1):
            action_list.append('DOWN')
        if (y > 0):
            action_list.append('LEFT')
        if (y < state.board.shape[1] - 1):
            action_list.append('RIGHT')
        return action_list
    
    def result(self, board, action):
        """ Returns the NPuzzle board that result from executing the given
        given action on the given NPuzzle board."""
        x, y = np.where(board.board == 0)
        new = board.board.copy()
        if (action == 'UP'):
            new[x, y], new[x-1, y] = new[x-1, y], new[x, y]
            return new
        if (action == 'DOWN'):
            new[x, y], new[x+1, y] = new[x+1, y], new[x, y]
            return new
        if (action == 'LEFT'):
            new[x, y], new[x, y-1] = new[x, y-1], new[x, y]
            return new
        if (action == 'RIGHT'):
            new[x, y], new[x, y+1] = new[x, y+1], new[x, y]
            return new

    def goal_test(self, board):
        """ Test if NPuzzle is at the end state [0,1,2,...,N-1]."""
        if np.array_equal(np.ravel(board.board), np.arange(board.board.size)):
            return True
        return False
      
    def path_cost(self, a, b, c, d):
        return 0
    
    def __repr__(self):
        return np.array2string(self.board)

class NPuzzleState(object):
 
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        #print("init state\n", self.state,"\n")
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    
    def neighbors(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = problem.result(self.state, action)
        return NPuzzleState(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                          action, next))     
    
    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, NPuzzleState) and self.state == other.state

    def __hash__(self):
        return hash(repr(self.state))
                #np.array2string(repr(self.state.board)))
        #return hash(self.__str__())
    
    #def __str__(self):
    #    #print('State is\n', self.state)#.board)
    #    #self.state
    #    return np.array2string(self.state)#.board)
    
    #def __contains(self, x)__:
        
       
              
class Solver:

    def __init__(self, initialBoard):
        self.nPuzzleState = NPuzzleState(initialBoard)
        #print(self.nPuzzleState.state.board)
        '''
        path_to_goal = 0
        cost_of_path = 0
        nodes_expanded = {}
        fringe_size = 0
        max_fringe_size = 0
        search_depth = 0
        max_search_depth = 0
        running_time = 0
#       max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss       
'''
    def _success(self, node):
        print('Success!')
        print(node.state.board)
        
    def _failure(self):
        print('Cannot find solution')
        return False

    def bfs(self):
        #return None #Success or Failure
        print('Running Breadth First Search...\n')
        frontier = Queue()
        frontier.enqueue(self.nPuzzleState)
        #frontier.enqueue(['1231245'])
        print('Frontier empty? ', frontier.empty())
        #print('Frontier List:', frontier)
        explored = set()
        print('Explored is', explored)
        
        while not frontier.empty():
            print('\nIn Loop...')
            node = frontier.dequeue()
            print(node.state)
            explored.add(node)#.state)
            #print(explored)
            print('\nAdded node to explored:\n')
            if node.state.goal_test(node.state):
                return self._success(node)
            print('\nafter goal_test')
            print('length of frontier', len(frontier))
            print('length of neighbor set', len(node.neighbors(node.state)))
            for neighbor in node.neighbors(node.state):
                #print(len(neighbor))
                print('\nlooping through neighbors list')
                
                print('Neighbor node is NOT in frontier', neighbor not in frontier)
                print('Neighbor node is NOT in explored', neighbor not in explored)
                if neighbor not in frontier and neighbor not in explored:
                    print('PROGRESSSSSS????')
                    #frontier.enqueue(neighbor)
            #print('Out of for loop')        
            #print('frontier length', len(frontier))
            #print(frontier.empty())
        
        print('wadafaaak')
        return self._failure
    
     
    
def run():
    parser = argparse.ArgumentParser(description = "n-Puzzle Game Search")
    parser.add_argument('method', nargs=1, type=str)
    parser.add_argument('input_board', nargs='+', type=lambda x:x.split(','))
    args = parser.parse_args()
    
    # reshaping board into a n by n array
    board = np.array(args.input_board)
    board = board.reshape(int(np.sqrt(board.size)), int(np.sqrt(board.size)))
        
    print('\nPlaying nPuzzle with Search method: ', args.method)
    print('Initial nPuzzle Board:\n', board)
    print('Size of board: ', board.size)
    #agent(board, method)
    
    
if __name__ == '__main__':
    #run()
    #t1 = NPuzzle(np.array([3,1,2,6,0,5,4,8,7]).reshape(3,3))
    #print('\nTest 1 board: \n', t1.board)
    #print(t1.actions(t1))
    #print(t1.result(t1,'RIGHT'))
    #t2 = NPuzzle(np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3))
    #print('\nTest 2 board: \n', t2.board)
    #print('\nTest 2 goal_test: \n', t2.goal_test(t2))
    #print(t2.actions(t2))
    t3 = NPuzzle(np.array([1,2,0,4,5,6,7,8,3]).reshape(3,3))
    #print('\nTest 3 board:\n', t3.board)
    #print(t3.actions(t3))
    bfs = Solver(t3)
    #print('break\n')
    bfs.bfs()
    
    
    
    #def dfs(initialState, goalTest):
    #    return None #Success or Failure
    #    
    #    frontier = stack.new(initialState)
    #    explored = Set.new()
    #    
    #    while not frontier.isEmpty():
    #        state = frontier.pop()
    #        explored.add(state)
    #        
    #        if goalTest(state):
    #            return success(state)
    #        
    #        for neighbors in state.neighbors():
    #            if neighbor not in frontier U explored:
    #                frontier.push(neighbor)
    #    
    #    
    #def aStar(initialState, goalTest):
    #    return False
    #    
    #def ida(initialState, goalTest):
    #    return False
    
