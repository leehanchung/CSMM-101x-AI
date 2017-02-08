# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:23:23 2017


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
import time
#import psutil, os
import numpy as np
#import queue as queue

    
class Queue:
    
    def __init__(self):
        self.A = []
        self.start = 0
            
    def empty(self):
        return not bool(len(self.A) - self.start)
    
    def extend(self, items):
        self.A.extend(items)              
           
    def enqueue(self, item):
        self.A.append(item)
    
    def dequeue(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e
    
    def __eq__(self, item):
        return set(item.A) == set(self.A)
    
    def __contains__(self, item):
        return item in self.A[self.start:]
    
    def __len__(self):
        return len(self.A) - self.start
    
    def __repr__(self):
        return str(self.A[self.start:])
    
class NPuzzle(object):
    
    """ Low-level physical configuration of N-Puzzle tiles."""
    
    def __init__(self, board):
        self.board = board
        
    def actions(self, state):
        """ Actions available in the current puzzle. Find the position of 
        zero first, and check if zero can be moved in up, down, left, right 
        directions. N-puzzle has edge and does not wrap around board. Returns 
        a list of available actions."""
        size = int(sqrt(len(state.board)))
        tmp = np.array(state.board).reshape(size, size)
        x, y = np.where(tmp == 0)
        action_list = []
        if (x > 0):
            action_list.append('UP')
        if (x < tmp.shape[0] - 1):
            action_list.append('DOWN')
        if (y > 0):
            action_list.append('LEFT')
        if (y < tmp.shape[1] - 1):
            action_list.append('RIGHT')
        return action_list
    
    def result(self, state, action):
        """ Returns the NPuzzle board that result from executing the given
        given action on the given NPuzzle board."""
        size = int(sqrt(len(state.board)))
        new = np.array(state.board).reshape(size, size)
        x, y = np.where(new == 0)
        if (action == 'UP'):
            new[x, y], new[x-1, y] = new[x-1, y], new[x, y]
            return NPuzzle(new.reshape(new.size).tolist())
        if (action == 'DOWN'):
            new[x, y], new[x+1, y] = new[x+1, y], new[x, y]
            return NPuzzle(new.reshape(new.size).tolist())
        if (action == 'LEFT'):
            new[x, y], new[x, y-1] = new[x, y-1], new[x, y]
            return NPuzzle(new.reshape(new.size).tolist())
        if (action == 'RIGHT'):
            new[x, y], new[x, y+1] = new[x, y+1], new[x, y]
            return NPuzzle(new.reshape(new.size).tolist())

    def goal_test(self, state):
        " Test if NPuzzle is at the end state [0,1,2,...,N-1]."
        if state.board == np.arange(len(state.board)).tolist():
            return True
        return False
      
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def __repr__(self):
        return str(self.board)
    

class NPuzzleState(object):
 
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
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
        print('hahiahiahi')
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        #print('path******************************')
        while node:
            path_back.append(node)
            node = node.parent
            #print(node.state.board)
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, NPuzzleState) and self.state == other.state

    def __hash__(self):
        return hash(str(self.state.board))           
        
    def __repr__(self):
        return str(self.state.board)
        
              
class Solver:

    def __init__(self, initialBoard):
        self.nPuzzleState = NPuzzleState(initialBoard)
        self.path_to_goal = []
        self.cost_of_path = 0
        self.nodes_expanded = 0
        self.fringe_size = 0
        self.max_fringe_size = 0
        self.search_depth = 0
        self.max_search_depth = 0
        self.running_time = 0
        self.max_ram_usage = 0 #resource.getrusage(resource.RUSAGE_SELF).ru_maxrss       

    def _success(self, node):
        print('\n\nSuccess!')
        print(node.solution)
        self.cost_of_path = node.path_cost
        self.search_depth = node.depth
        while node.parent is not None:
            self.path_to_goal.insert(0, node.action)
            node = node.parent
        self.running_time = time.time() - self.running_time
        print('path_to_goal', self.path_to_goal)    
        print('cost_of_path', self.cost_of_path)
        print('nodes_expended:', self.nodes_expanded)
        print('fringe_size:', self.fringe_size)
        print('max_fringe_size:', self.max_fringe_size)
        print('search_depth', self.search_depth)
        print('max_search_depth', self.max_search_depth)
        print('running_time', self.running_time)
        print('max_ram_usage', self.max_ram_usage)
        
    def _failure(self):
        print('Cannot find solution')
        return False

    def bfs(self):
        self.running_time = time.time()
        frontier = Queue()
        frontier.enqueue(self.nPuzzleState)
        self.fringe_size += 1
        explored = set()
        
        while not frontier.empty():
            print('\n\n* * * * *  Looping  * * * * * ')
            node = frontier.dequeue()
           
            self.fringe_size -= 1
            #print('node object?\n', node)
            print('Exploring:', node.state)
            print('frontier', frontier)
            #print(node)
            explored.add(hash(node))#hash(str(node.state.board)))
            
            print(hash(node) == hash(str(node.state.board)))
            #print('\nNode added to explored set()\n')
            if node.state.goal_test(node.state):
                print('checking goal test')
                return self._success(node)
            self.nodes_expanded += 1
            #print('after goal_test')
            #print('length of frontier', len(frontier))
            
            for neighbor in node.neighbors(node.state):
                print('\nChecking neighbor:', neighbor.state)
                #print('\n...Looping through neighbors list...', len(node.neighbors(node.state)))
                print('Neighbor node is NOT in frontier:', neighbor not in frontier)
                print('Neighbor node is NOT in explored:', hash(neighbor) not in explored) #hash(str(neighbor.state.board)) not in explored)
                #print('frontier length', len(frontier))
                #print('neighbors object?\n', neighbor)
                if neighbor not in frontier and hash(neighbor) not in explored:
                    print('Adding to frontier', neighbor.state)
                    #print(explored)
                    frontier.enqueue(neighbor)
                    self.fringe_size += 1
                    if neighbor.depth > self.max_search_depth:
                        self.max_search_depth = neighbor.depth
                    
                #print('frontier length', len(frontier))
            print('Frontier is now:', frontier)
        
            print('wadafaaak')
            print(len(frontier))
            print(len(frontier.A))
            print('max_fringe_size', self.max_fringe_size)
            print(self.nodes_expanded)
            if self.fringe_size > self.max_fringe_size:
                        self.max_fringe_size = self.fringe_size
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
    t = NPuzzle([1,2,5,3,4,0,6,7,8])#NPuzzle(np.array([1,2,0,4,5,6,7,8,3]).reshape(3,3))
    #t3 = NPuzzle([3,1,2,0,4,5,6,7,8])
    #print('\nTest 3 board:\n', t3.board)
    #print(t3.actions(t3))
    bfs = Solver(t)
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
    
