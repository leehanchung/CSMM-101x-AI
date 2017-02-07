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
import numpy as np
import pandas as pd

'''
class nPuzzleBoard(state, action):
    
    def __init__(self):
        return
    
    
class nPuzzleState(action):
    
    def __init__(self):
        return
    
    
class nPuzzleAction(state):
    
    def __init__(self):
        return
              
def runNPuzzle():
    
    def BFS(initialState, goalTest):
        return None #Success of Failure
        frontier = queue.new(initialState)
        explored = Set.new()
        
        while not frontier.isEmpty():
            state = frontier.dequeue()
            explore.add(state)
            
            if goal(test(state):
                return sucess(state)
            
            if neighbor in state.neighbors():
                if neighbor not in frontier explored:
                    frontier.enqueue(neighbor)
                    
        return failure
'''    
    
def run():
    """
    parse commands 
    """
    #from optparse import OptionParser
    #usgStr = """
    #USAGE:     asdf
    #EXAMPLES:  asdf
    #"""
    parser = argparse.ArgumentParser(description = "n-Puzzle Game Search")
    parser.add_argument('method', help="Search method")
    print("wtf\n")
    
    
    
if __name__ == '__main__':
    run()