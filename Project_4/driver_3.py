# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:22:36 2017

ColumbiaX CSMM.101x Artificial Intelligence Project 4

Sudoku solver with Constraint Satisfication Problem method, done with 
backtracking search with minimum-remaining-value and forward checking. Also 
applies AC3 to reduce search space in every iteration.

Usage:
python driver_3.py <sudoku puzzle>
<sudoku puzzle> is represented as a single line string of 81 digits, with 0 
representing an empty box.

Output:
<output.txt> contained solved puzzle in a single line string. In addition, 
on-screen print-out for the initial input puzzle and the solved puzzle
"""
import argparse, copy

class Sudoku:
    """ 
    Sudoku as a Constraint Satisfication Problem.
        variables -> boxes, 
        domain -> digits, 
        constraints -> all diff within units
        neighbors -> peers
    """
    def __init__(self, str_sudoku):
        self.rows = 'ABCDEFGHI'
        self.cols = '123456789'
        self.digits = '123456789'
        def cross(A, B):
            return [a+b for a in A for b in B]
        self.boxes = cross(self.rows, self.cols)
        row_units = [cross(r, self.cols) for r in self.rows]
        col_units = [cross(self.rows, c) for c in self.cols]
        square_units = [cross(r, c) for r in ('ABC','DEF','GHI') for c in ('123','456','789')]    
        self.unitlist = row_units + col_units + square_units
        self.units = dict((box, [u for u in self.unitlist if box in u]) for box in self.boxes)
        self.peers = dict((b, set(sum(self.units[b],[]))-set([b])) for b in self.boxes)
        self.sudoku = self.set_sudoku(str_sudoku)        
        
    def set_sudoku(self, sudoku):
        """ 
        Takes a sudoku puzzle string and return a sudoku puzzle set of 
        {box: possible digits of the box} 
        """
        values = []        
        for d in sudoku:
            if d == '0':
                values.append(self.digits)
            elif d in self.digits:
                values.append(d)
        assert len(values) == 81
        return dict(zip(self.boxes, values))
    
    def constraints(self, box_i, i, box_j, j):
        return (i != j)

    def display(self):
        " display a sudoku set "
        width = 1+max(len(self.sudoku[b]) for b in self.boxes)
        line = '+'.join(['-'*(width*3)]*3)
        print('\n')
        for r in self.rows:
            print(''.join(self.sudoku[r+c].center(width)+('|' if c in '36' else '')
                          for c in self.cols))
            if r in 'CF': print(line)
        return None
    
    def prune(self, box, digit):
        " remove digit from a sudoku box "
        self.sudoku[box] = self.sudoku[box].replace(digit, '')
        
    def goal_test(self):
        return True if all(len(self.sudoku[b]) == 1 for b in self.boxes) else False
        
    def tostring(self):
        " returns sudoku puzzle in a single line string of 81 digits " 
        values = ''
        for box in self.boxes:
            if len(self.sudoku[box]) > 1:
                values += '0'
            elif len(self.sudoku[box]) == 1:
                values += self.sudoku[box]
        return values


def AC3(Sudoku):
    " AC3 constraint satification for reducing puzzle from known constraints "
    queue = [(Xi, Xk) for Xi in Sudoku.boxes for Xk in Sudoku.peers[Xi]]
    while queue:
        (Xi, Xj) = queue.pop()
        if revise(Sudoku, Xi, Xj):
            if not Sudoku.sudoku[Xi]:
                return False
            for Xk in Sudoku.peers[Xi]:
                if Xk != Xi:
                    queue.append((Xk, Xi))
    return True

def revise(Sudoku, Xi, Xj):
    """Return true if we remove a value."""
    revised = False
    for x in Sudoku.sudoku[Xi]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if all(not Sudoku.constraints(Xi, x, Xj, y) for y in Sudoku.sudoku[Xj]):
            Sudoku.prune(Xi, x)
            revised = True
    return revised

def mrv(Sudoku):
    """Minimum-remaining-values heuristic. Choose variables with fewest legal
    values in its domain """
    n, s = min((len(Sudoku.sudoku[b]), b) for b in Sudoku.boxes if len(Sudoku.sudoku[b]) > 1)
    return n, s

def forward_checking(Sudoku, box, digit):
    """Prune peer digits inconsistent with box = digit."""
    for peer in Sudoku.peers[box]:
        if len(Sudoku.sudoku[peer]) == 1 and digit == Sudoku.sudoku[peer]:
            " fail if other cells of length 1 already contains digit "
            return False
        else:
            " prune digit from box' peers"
            Sudoku.prune(peer, digit)
            if len(Sudoku.sudoku[peer]) == 0:
                return False
    return True
    
def backtracking_search(Sudoku):
    """ 
    Apply constraints with AC3 first to reduce search space. 
    Not using AIMA's python structure because the Sudoku class is already 
    setup as a python set similar to AIMA's assignment variable.
    """
    if not AC3(Sudoku):
        return False 
    if Sudoku.goal_test(): 
        return Sudoku 
    """ 
    Apply minimum-remaining-values heuristics to pick variables with fewest
    legal values in its domain
    """ 
    _, box = mrv(Sudoku)
    for digit in Sudoku.sudoku[box]:
        new_sudoku = copy.deepcopy(Sudoku)
        " forward checking digits, if valid move, proceed to go deeper "
        if forward_checking(new_sudoku, box, digit):
            new_sudoku.sudoku[box] = digit
            attempt = backtracking_search(new_sudoku)
            if attempt:
                return attempt    
            
def read_args():
    " Read arguments from command prompt. Does not enforce any check"
    parser = argparse.ArgumentParser(description = "sudoku solver")
    parser.add_argument('input_board', nargs='+', type=lambda x:x.split(','))
    args = parser.parse_args()
    return args.input_board[0][0]

def run_test():
    """ 
    testing AC3 or backtracking_search over all 400 test cases contained in
    sudokus_start.txt and verify output against solutions contained in
    sudokus_finish.txt.
    """
    with open('sudokus_start.txt') as start, open('sudokus_finish.txt') as finish:
        count = 0
        num_sudoku = 0
        for line in start:
            num_sudoku += 1
            sudoku = Sudoku(line)            
            sudoku.display()
            
            " Testing AC3 constraint propagation "
            """
            AC3(sudoku)
            if str(sudoku.tostring()+'\n') == str(finish.readline()):
                count +=1
                sudoku.display()
                print('\n', num_sudoku, count)
            """
            
            " Testing backtracking_search "
            #"""
            bts = backtracking_search(sudoku)            
            if str(bts.tostring()+'\n') == str(finish.readline()):
                count +=1
                bts.display()
                print('\nTest#', num_sudoku, ' Success#',count)
            #"""
    
if __name__ == "__main__":
    """
    sudoku = Sudoku(read_args())
    sudoku.display()
    #print(sudoku.units['A1'])
    sudoku = backtracking_search(sudoku)
    sudoku.display()
    f = open('output.txt', 'w')
    f.write(sudoku.tostring())
    f.close()
    """
   
    #sudoku = Sudoku('000100702030950000001002003590000301020000070703000098800200100000085060605009000')
    #sudoku.display()
    #print(sudoku.sudoku)
    #AC3(sudoku)
    #sudoku.display()
    #print(sudoku.sudoku)
    
    #bts = backtracking_search(sudoku)
    #print(bts)
    #bts.display()
    #print(sudoku.tostring(),'\n')
    
    run_test()
    