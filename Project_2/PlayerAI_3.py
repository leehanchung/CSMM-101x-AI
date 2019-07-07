# -*- coding: utf-8 -*-
"""
Python 3.5

ColumbiaX CSMM.101x Artificial Intelligence Project 2


PlayerAI for 2048 game, solved using minimax with alpha-beta pruning and 
depth and time limited search. Search time is limited by 0.1 seconds per
project spec. Max search depth limit is inverse proportional to the number of 
empty cells on the grid. Move ordering is favoring top and left. Heuristics is 
a combination of monotonicity, number of empty cells, largest tile, and bias 
towards largest tile being in the corner cells.

Project score 200/200, with 60% in 1024, 36% in 2048, and 4% in 4096.
"""
import time
from math import inf
from BaseAI_3 import BaseAI

class PlayerAI(BaseAI):
    
    def __init__(self):
        self.max_search_depth= 0
        self.depth = 0
        self.start_time = 0
    
    def getMove(self, grid):
        self.max_search_depth = self.searchDepth(grid)
        self.alpha = -inf
        self.beta = inf
        self.depth = 0
        self.start_time = time.clock()
        move, _ = self.maximize(grid, -inf, inf)
        return move
    
    def maximize(self, grid, alpha, beta):
        self.depth += 1
        if self.depth >= self.max_search_depth:
            return None, self.heuristic(grid)        
        elif not grid.canMove():
            return None, self.heuristic(grid)        
        elif time.clock() - self.start_time >= 0.099:
            return None, self.heuristic(grid)        
            
        max_move, max_utility = None, -inf
        " Setting move search preference to prioritize UP and LEFT. "
        reordered_moves = grid.getAvailableMoves()
        reordered_moves = sorted(map(int, reordered_moves), key=lambda num: num%2)

        for move in reordered_moves:            
            gridCopy = grid.clone()
            gridCopy.move(move)
            _, utility = self.minimize(gridCopy, alpha, beta)
            self.depth -=1
            
            if utility > max_utility:
                max_move, max_utility = move, utility
                
            if max_utility >= beta:
                break
            
            if max_utility > alpha:
                alpha = max_utility
            
            if time.clock()- self.start_time >= 0.1:
                break 
        return max_move, max_utility

    def minimize(self, grid, alpha, beta):
        self.depth += 1       
        if self.depth >= self.max_search_depth:
            return None, self.heuristic(grid)
        elif not grid.canMove():
            return None, self.heuristic(grid)        
        elif time.clock() - self.start_time >= 0.1:
            return None, self.heuristic(grid)        
        
        min_cell, min_utility = None, inf
        for cell in grid.getAvailableCells():
            gridCopy = grid.clone()
            " using 2 for now, since it has 90% chance "
            gridCopy.setCellValue(cell, 2)                                       
            _, utility = self.maximize(gridCopy, alpha, beta)
            self.depth -=1
            
            if utility < min_utility:
                min_cell, min_utility = cell, utility
            
            if min_utility <= alpha:
                break
            
            if min_utility < beta:
                beta = min_utility
        
            if time.clock()- self.start_time >= 0.1:
                break
        return min_cell, min_utility
    
    def heuristic(self, grid):
        """ Replace utility function with evaluation function to estimate value
        of current board configurations. eval(s) is a heuristic at state s """
        #h1 = self.gridValue(grid)
        h_monotonicity = self.h_monotonicity(grid) / 8
        h_empty = len(grid.getAvailableCells()) / 16
        h_max_multiplier = grid.getMaxTile() / 1024
        if grid.getMaxTile() == grid.map[0][0]:
            h_max_corner_multiplier = 2
        else: 
            h_max_corner_multiplier = 1
        return h_monotonicity * h_empty * h_max_multiplier * h_max_corner_multiplier
                
    def searchDepth(self, grid):
        """ Heuristic function that limits search depth depending on the number
        of empty cells on the 2048 grid. The higher number of empty cells, the
        less depth the search is, and vice versa. """
        empty_cells = len(grid.getAvailableCells())
        if empty_cells >= 16: 
            return 2
        elif empty_cells >= 8:
            return 4
        elif empty_cells >= 4:
            return 6
        else:
            return 8
    
    def h_monotonicity(self, grid):
        """ Heuristic function that scores the grid based on the tile ordering
        is entirely decreasing. """        
        n = grid.map
        m = list(map(list, zip(*n)))
        h = 0
        for i in range(len(n)):
            if all(earlier >= later for earlier, later in zip(n[i], n[i][1:])):
                h += 1
            if all(earlier >= later for earlier, later in zip(m[i], m[i][1:])):
                h += 1
        return h