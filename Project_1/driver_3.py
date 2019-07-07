# -*- coding: utf-8 -*-
"""
Python 3.5

ColumbiaX CSMM.101x Artificial Intelligence Project 1

Usage: python driver_3.py <method> <board>
<method>: bfs, dfs, ast, or ida
<board>:  board of size N^2 seoarated by comma, with 0 represent blank space
          for example for 8-Puzzle, 1,2,5,3,4,0,6,7,8
          
Does not do error check w/ incorrect method name or incorrect board size.
Pass 5/5 test cases on Vocareum. scored 200/200.
max_ram_usage and running_time not scored.
"""
import argparse
import time
import sys
from math import sqrt
from collections import deque
from heapq import heappush, heappop, heapify

""" setting keys for a* heuristic comparisons """
action_dict = {'Up':1, 'Down':2, 'Left':3, 'Right':4}
    
class NPuzzle(object):
    
    """ Low-level physical configuration of N-Puzzle tiles. Board represented
    using tuple so its hashable."""
    
    def __init__(self, board):
        self.board = board
        
    def actions(self, state):
        """ Actions available in the current puzzle. Find the position of 
        zero first, and check if zero can be moved in up, down, left, right 
        directions. N-puzzle has edge and does not wrap around board. Returns 
        a list of available actions."""
        size = int(sqrt(len(state.board)))
        x = self.board.index(0)
        action_list = []
        
        if x > size-1:
            action_list.append('Up')
        if x < len(self.board)-size:
            action_list.append('Down')
        if x % size != 0:
            action_list.append('Left')
        if x % size != size-1:
            action_list.append('Right')
        return action_list
    
    def reverse_action(self,state):
        """ adding actions in reverse order. Used for depth first search as 
        speficied by project """
        size = int(sqrt(len(state.board)))
        x = self.board.index(0)
        action_list = []
        
        if x % size != size-1:
            action_list.append('Right')
        if x % size != 0:
            action_list.append('Left')
        if x < len(self.board)-size:
            action_list.append('Down')
        if x > size-1:
            action_list.append('Up')
        return action_list
    
    def result(self, state, action):
        """ Returns the NPuzzle board that result from executing the given
        given action on the given NPuzzle board. Have to change to list()
        because tuple() is immutable """
        size = int(sqrt(len(state.board)))
        new = list(state.board)
        x = state.board.index(0)
        if (action == 'Up'):
            new[x], new[x-size] = new[x-size], new[x]
            return NPuzzle(tuple(new))
        if (action == 'Down'):
            new[x], new[x+size] = new[x+size], new[x]
            return NPuzzle(tuple(new))
        if (action == 'Left'):
            new[x], new[x-1] = new[x-1], new[x]
            return NPuzzle(tuple(new))
        if (action == 'Right'):
            new[x], new[x+1] = new[x+1], new[x]
            return NPuzzle(tuple(new))

    def goal_test(self, state, goal):
        return state.board == goal
      
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def __repr__(self):
        return str(self.board)
    
    def __eq__(self, other):
        return self.board == other.board and isinstance(other, NPuzzle)
    

class NPuzzleState(object):
    
    """ NPuzzleState represents a node in the NPuzzle game search tree. It
    contains an NPuzzle object, parent of the node, action taken to get to the
    node, and path_cost and search depth of the node. Hash value is based on
    the board within the NPuzzle object only to prevent adding duplicates nodes
    with different parents/actions during search."""
 
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        """ design decision to calculate manhattan distance heuristics at the 
        time of object creation to reduce repeated calculation for A* and 
        iterative depth A*. It slows down BFS/DFS by a few seconds but saves
        a lot of time for AST and IDA """
        self.f = self.path_cost + self.manhattan()
    
    def neighbors(self):
        "List the nodes reachable in one step from this node in UDLR order."
        x = []
        for action in self.state.actions(self.state):
            x.append(NPuzzleState(self.state.result(self.state, action), 
                                  self, 
                                  action, 
                                  self.state.path_cost(self.path_cost, 
                                                       self.state,
                                                       action, 
                                                       next)))
        return x
    
    def reverse_neighbors(self):
        "List the nodes reachable in one step from this node, in reverse UDLR"
        x = []
        for action in self.state.reverse_action(self.state):
            x.append(NPuzzleState(self.state.result(self.state, action), 
                                  self, 
                                  action, 
                                  self.state.path_cost(self.path_cost, 
                                                       self.state,
                                                       action, 
                                                       next)))
        return x
    
    def manhattan(self):
        " Calculate the manhattan distance of the board, for heuristics "
        w = int(sqrt(len(self.state.board)))
        return sum((abs(i//w - self.state.board.index(i)//w) + 
                    abs(i%w - self.state.board.index(i)%w) 
                    for i in self.state.board if i != 0))
        
    def __eq__(self, other):
        return self.state.board == other.state.board and isinstance(other, NPuzzleState)

    def __hash__(self): 
        return hash(str(self.state.board))
        
    def __repr__(self):
        return str(self.state.board)
    
    def __lt__(self, other):
        """ For heapq comparisons. 1) compare f(n) = g(n) + h(n) with h(n) being
            the manhattan() distance, 2) compare directions in the order of
            UP, DOWN, LEFT, RIGHT. 3) To compensate for the same f(n) and same 
            action, I've choose to go wide in A* by choosing higher g(n)"""
        if self.f < other.f:
            return True
        elif self.f == other.f:
            if action_dict[self.action] < action_dict[other.action]:
                return True
            elif self.action == other.action:
                return self.depth > other.depth
            return False
        else:
            return False
        
              
class Solver:
    
    """ Solver class contains four different algorithm to search: bfs() -
    breadth first search, dfs() - depth first search, ast() - A* search, and
    ida() - iterative depth A* search. Also tracks the stats of search 
    algorithms """

    def __init__(self, initialBoard):
        self.nPuzzleState = NPuzzleState(initialBoard)
        self.path_to_goal = []
        self.cost_of_path = 0
        self.nodes_expanded = 0
        self.fringe_size = 0
        self.max_fringe_size = 0
        self.search_depth = 0
        self.max_search_depth = 0
        self.goal = tuple(range(0,len(initialBoard.board)))

    def __success(self, node):
        self.cost_of_path = node.path_cost
        self.search_depth = node.depth
        " work backwards from answer node to get the path solution " 
        while node.parent is not None:
            self.path_to_goal.insert(0, node.action)
            node = node.parent       
        return None
        
    def __failure(self):
        print('Cannot find solution')
        return None

    def bfs(self):
        """ using deque() as FIFO Queue, right side in, left side out"""
        frontier = deque()
        frontier.append(self.nPuzzleState)
        self.fringe_size += 1
        """ using a set() to track both frontier and explored nodes"""
        frontier_U_explored = set()
        frontier_U_explored.add(self.nPuzzleState)
        
        while frontier:
            node = frontier.popleft()
            self.fringe_size -= 1
            
            if node.state.goal_test(node.state, self.goal):
                return self.__success(node)
            self.nodes_expanded += 1
            
            for neighbor in node.neighbors():
                if neighbor not in frontier_U_explored:
                    """ enqueue in UDLR order, the order of how actions are
                    checked at Class NPuzzle. """
                    frontier_U_explored.add(neighbor)
                    frontier.append(neighbor)
                    self.fringe_size += 1
                    if neighbor.depth > self.max_search_depth:
                        self.max_search_depth = neighbor.depth

            if self.fringe_size > self.max_fringe_size:
                self.max_fringe_size = self.fringe_size
            
        return self.__failure
    
    def dfs(self):
        """ using deque() as LIFO stack, right side in/out"""
        frontier = deque()
        frontier.append(self.nPuzzleState)
        self.fringe_size += 1
        """ using a set() to track both frontier and explored nodes"""
        frontier_U_explored = set()
        frontier_U_explored.add(self.nPuzzleState)
        
        while frontier:
            node = frontier.pop()
            self.fringe_size -= 1
            
            if node.state.goal_test(node.state, self.goal):
                return self.__success(node)
            self.nodes_expanded += 1            
            
            for neighbor in node.reverse_neighbors():
                if neighbor not in frontier_U_explored:
                    """ enqueue in *reverse* UDLR order, the order of how 
                    actions are checked at Class NPuzzle. """
                    #reverse_frontier.appendleft(neighbor)
                    frontier.append(neighbor)
                    frontier_U_explored.add(neighbor)
                    self.fringe_size += 1
                    if neighbor.depth > self.max_search_depth:
                        self.max_search_depth = neighbor.depth
                        
            if self.fringe_size > self.max_fringe_size:
                self.max_fringe_size = self.fringe_size
                
        return self.__failure    
    
    def astar(self, f_limit=sys.maxsize):
        """ A* search used by both ast() and ida(); default cost f(n) is set
        to maxsize for ast() """
        frontier = []
        """ priority queue using heapq, priorities cost function f(n) is 
        calculated at nPuzzleState object as nPuzzleState.f """
        heappush(frontier, self.nPuzzleState)
        self.fringe_size += 1
        frontier_U_explored = set()
        frontier_set = set()
        frontier_set.add(self.nPuzzleState)
        d = f_limit
        
        while frontier:
            node = heappop(frontier)
            self.fringe_size -= 1
            frontier_U_explored.add(self.nPuzzleState)
            frontier_set.remove(node)
            
            if node.state.goal_test(node.state, self.goal):
                return node
            self.nodes_expanded += 1          
            
            for neighbor in node.neighbors():
                if neighbor not in frontier_U_explored:
                    if neighbor.f < d:
                        heappush(frontier, neighbor)
                        frontier_U_explored.add(neighbor)
                        frontier_set.add(neighbor)
                        self.fringe_size += 1
                        if neighbor.depth > self.max_search_depth:
                            self.max_search_depth = neighbor.depth
                elif neighbor in frontier_set:
                    " reset neighbor weight if duplicate found in frontier "
                    frontier.remove(neighbor)
                    heappush(frontier, neighbor)
                    heapify(frontier)
                    
            if self.fringe_size > self.max_fringe_size:
                self.max_fringe_size = self.fringe_size
        return False

    def ast(self):
        result = self.astar()
        if result:
            return self.__success(result)
        return self.__failure
        
    def ida(self):
        f_limit = self.nPuzzleState.f
        while True:
            self.nodes_expanded = 0
            result = self.astar(f_limit)
            if result:
                return self.__success(result)
            else:
                f_limit += int(sqrt(f_limit))
                
    
def run():
    parser = argparse.ArgumentParser(description = "n-Puzzle Game Search")
    parser.add_argument('method', nargs=1, type=str)
    parser.add_argument('input_board', nargs='+', type=lambda x:x.split(','))
    args = parser.parse_args()
    
    " args.input_board reads into a list of list, reformat to tuple"
    board = args.input_board
    board = tuple([int(i) for i in board[0]])
    game = NPuzzle(board)
    print('\n* * * * * * * * * * * * * * * *\nInitial nPuzzle Board:\n', game)
    solution = Solver(game)
    
    running_time = time.time()
    if args.method == ['bfs']:
        solution.bfs()
    elif args.method == ['dfs']:
        solution.dfs()
    elif args.method == ['ast']:
        solution.ast()
    elif args.method == ['ida']:
        solution.ida()
    running_time = time.time() - running_time
    " Anaconda Python Windows 10 doesnt have memory tracking "
    max_ram_usage = 1337 

    " On screen print out solutions "
    print('\n - - solution - - \n')                            
    print('path_to_goal', solution.path_to_goal)    
    print('cost_of_path', solution.cost_of_path)
    print('nodes_expended:', solution.nodes_expanded)
    print('fringe_size:', solution.fringe_size)
    print('max_fringe_size:', solution.max_fringe_size)
    print('search_depth', solution.search_depth)
    print('max_search_depth', solution.max_search_depth)
    print('running_time:', running_time)
    print('max_ram_usage', max_ram_usage)

    " printing to output.txt according to project spec."                                
    with open('output.txt', 'w') as f:
        f.write('path_to_goal:' + str(solution.path_to_goal) +'\n')
        f.write('cost_of_path:' + str(solution.cost_of_path) +'\n')
        f.write('nodes_expanded:' + str(solution.nodes_expanded) + '\n')
        f.write('fringe_size:' + str(solution.fringe_size) + '\n')
        f.write('max_fringe_size:' + str(solution.max_fringe_size) + '\n')
        f.write('search_depth:' + str(solution.search_depth) + '\n')
        f.write('max_search_depth:' + str(solution.max_search_depth) + '\n')
        f.write('running_time:' + str(running_time) + '\n')
        f.write('max_ram_usage:' + str(max_ram_usage) + '\n')
        
    
if __name__ == '__main__':
    run()