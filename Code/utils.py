"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""


# header files
import numpy as np
import cv2
from heapq import heappush, heappop


# class for AStar
class AStar(object):
    # init function
    def __init__(self, start, goal, clearance, radius, stepSize):
        self.start = start
        self.goal = goal
        self.numRows = 200
        self.numCols = 300
        self.stepSize = stepSize
        self.clearance = clearance
        self.radius = radius
        self.graph = {}
        self.distance = {}
        self.path = {}
        self.costToCome = {}
        self.costToGo = {}
        self.visited = {}
        
        for row in range(1, self.numRows + 1):
            for col in range(1, self.numCols + 1):
                self.visited[(row, col)] = False
                self.path[(row, col)] = -1
                self.graph[(row, col)] = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
                self.costToCome[(row, col)] = float('inf')
                self.costToGo[(row, col)] = float('inf')
                self.distance[(row, col)] = float('inf')
    
    # move is valid 
    def IsValid(self, currRow, currCol):
        return (currRow >= (1 + self.radius + self.clearance) and currRow <= (self.numRows - self.radius - self.clearance) and currCol >= (1 + self.radius + self.clearance) and currCol <= (self.numCols - self.radius - self.clearance))
    
    # checks for an obstacle
    def IsObstacle(self, row, col):
        # constants
        sum_of_c_and_r = self.clearance + self.radius
        sqrt_of_c_and_r = 1.4142 * sum_of_c_and_r
        
        # check circle
        dist1 = ((row - 150) * (row - 150) + (col - 225) * (col - 225)) - ((25 + sum_of_c_and_r) * (25 + sum_of_c_and_r))
        
        # check eclipse
        dist2 = ((((row - 100) * (row - 100)) / ((20 + sum_of_c_and_r) * (20 + sum_of_c_and_r))) + (((col - 150) * (col - 150)) / ((40 + sum_of_c_and_r) * (40 + sum_of_c_and_r)))) - 1
        
        # check triangles
        (x1, y1) = (120 - (2.62 * sum_of_c_and_r), 20 - (1.205 * sum_of_c_and_r))
        (x2, y2) = (150 - sqrt_of_c_and_r, 50)
        (x3, y3) = (185 + sum_of_c_and_r, 25 - (sum_of_c_and_r * 0.9247))
        first = ((col - y1) * (x2 - x1)) - ((y2 - y1) * (row - x1))
        second = ((col - y2) * (x3 - x2)) - ((y3 - y2) * (row - x2))
        third = ((col - y3) * (x1 - x3)) - ((y1 - y3) * (row - x3))
        dist3 = 1
        if(first <= 0 and second <= 0 and third <= 0):
            dist3 = 0
            
        (x1, y1) = (150 - sqrt_of_c_and_r, 50)
        (x2, y2) = (185 + sum_of_c_and_r, 25 - (sum_of_c_and_r * 0.9247))
        (x3, y3) = (185 + sum_of_c_and_r, 75 + (sum_of_c_and_r * 0.714))
        first = ((col - y1) * (x2 - x1)) - ((y2 - y1) * (row - x1))
        second = ((col - y2) * (x3 - x2)) - ((y3 - y2) * (row - x2))
        third = ((col - y3) * (x1 - x3)) - ((y1 - y3) * (row - x3))
        dist4 = 1
        if(first >= 0 and second >= 0 and third >= 0):
            dist4 = 0
        
        # check rhombus
        (x1, y1) = (10 - sqrt_of_c_and_r, 225)
        (x2, y2) = (25, 200 - sqrt_of_c_and_r)
        (x3, y3) = (40 + sqrt_of_c_and_r, 225)
        (x4, y4) = (25, 250 + sqrt_of_c_and_r)
        first = ((col - y1) * (x2 - x1)) - ((y2 - y1) * (row - x1))
        second = ((col - y2) * (x3 - x2)) - ((y3 - y2) * (row - x2))
        third = ((col - y3) * (x4 - x3)) - ((y4 - y3) * (row - x3))
        fourth = ((col - y4) * (x1 - x4)) - ((y1 - y4) * (row - x4))
        dist5 = 1
        dist6 = 1
        if(first >= 0 and second >= 0 and third >= 0 and fourth >= 0):
            dist5 = 0
            dist6 = 0
        
        # check square
        (x1, y1) = (150 - sqrt_of_c_and_r, 50)
        (x2, y2) = (120 - sqrt_of_c_and_r, 75)
        (x3, y3) = (150, 100 + sqrt_of_c_and_r)
        (x4, y4) = (185 + sum_of_c_and_r, 75 + (sum_of_c_and_r * 0.714))
        first = ((col - y1) * (x2 - x1)) - ((y2 - y1) * (row - x1))
        second = ((col - y2) * (x3 - x2)) - ((y3 - y2) * (row - x2))
        third = ((col - y3) * (x4 - x3)) - ((y4 - y3) * (row - x3))
        fourth = ((col - y4) * (x1 - x4)) - ((y1 - y4) * (row - x4))
        dist7 = 1
        dist8 = 1
        if(first <= 0 and second <= 0 and third <= 0 and fourth <= 0):
            dist7 = 0
            dist8 = 0
        
        # check rod
        first = ((col - 95) * (8.66 + sqrt_of_c_and_r)) - ((5 + sqrt_of_c_and_r) * (row - 30 + sqrt_of_c_and_r))
        second = ((col - 95) * (37.5 + sqrt_of_c_and_r)) - ((-64.95 - sqrt_of_c_and_r) * (row - 30 + sqrt_of_c_and_r))
        third = ((col - 30.05 + sqrt_of_c_and_r) * (8.65 + sqrt_of_c_and_r)) - ((5.45 + sqrt_of_c_and_r) * (row - 67.5))
        fourth = ((col - 35.5) * (-37.49 - sqrt_of_c_and_r)) - ((64.5 + sqrt_of_c_and_r) * (row - 76.15 - sqrt_of_c_and_r))
        dist9 = 1
        dist10 = 1
        if(first <= 0 and second >= 0 and third >= 0 and fourth >= 0):
            dist9 = 0
            dist10 = 0
        
        if(dist1 <= 0 or dist2 <= 0 or dist3 == 0 or dist4 == 0 or dist5 == 0 or dist6 == 0 or dist7 == 0 or dist8 == 0 or dist9 == 0 or dist10 == 0):
            return True
        return False
    
    # action move left
    def ActionMoveLeft(self, currRow, currCol):
        if(self.IsValid(currRow, currCol - self.stepSize) and self.IsObstacle(currRow, currCol - self.stepSize) == False and self.visited[(currRow, currCol - self.stepSize)] == False):
            return True
        return False

    # action move right
    def ActionMoveRight(self, currRow, currCol):
        if(self.IsValid(currRow, currCol + self.stepSize) and self.IsObstacle(currRow, currCol + self.stepSize) == False and self.visited[(currRow, currCol + self.stepSize)] == False):
            return True
        return False

    # action move up
    def ActionMoveUp(self, currRow, currCol):
        if(self.IsValid(currRow - self.stepSize, currCol) and self.IsObstacle(currRow - self.stepSize, currCol) == False and self.visited[(currRow - self.stepSize, currCol)] == False):
            return True
        return False

    # action move down
    def ActionMoveDown(self, currRow, currCol):
        if(self.IsValid(currRow + self.stepSize, currCol) and self.IsObstacle(currRow + self.stepSize, currCol) == False and self.visited[(currRow + self.stepSize, currCol)] == False):
            return True
        return False

    # action move right up
    def ActionMoveRightUp(self, currRow, currCol):
        if(self.IsValid(currRow - self.stepSize, currCol + self.stepSize) and self.IsObstacle(currRow - self.stepSize, currCol + self.stepSize) == False and self.visited[(currRow - self.stepSize, currCol + self.stepSize)] == False):
            return True
        return False

    # action move right down
    def ActionMoveRightDown(self, currRow, currCol):
        if(self.IsValid(currRow + self.stepSize, currCol + self.stepSize) and self.IsObstacle(currRow + self.stepSize, currCol + self.stepSize) == False and self.visited[(currRow + self.stepSize, currCol + self.stepSize)] == False):
            return True
        return False

    # action move left down
    def ActionMoveLeftDown(self, currRow, currCol):
        if(self.IsValid(currRow + self.stepSize, currCol - self.stepSize) and self.IsObstacle(currRow + self.stepSize, currCol - self.stepSize) == False and self.visited[(currRow + self.stepSize, currCol - self.stepSize)] == False):
            return True
        return False

    # action move left up
    def ActionMoveLeftUp(self, currRow, currCol):
        if(self.IsValid(currRow - self.stepSize, currCol - self.stepSize) and self.IsObstacle(currRow - self.stepSize, currCol - self.stepSize) == False and self.visited[(currRow - self.stepSize, currCol - self.stepSize)] == False):
            return True
        return False
    
    # update action
    def UpdateAction(self, currentNode, weight, newRow, newCol):
        new_cost_to_come = self.costToCome[currentNode] + weight
        new_cost_to_go = self.euc_heuristic(newRow, newCol)
        new_distance = new_cost_to_come + new_cost_to_go
                
        if(self.distance[(newRow, newCol)] > new_distance):
            self.distance[(newRow, newCol)] = new_distance
            self.costToCome[(newRow, newCol)] = new_cost_to_come
            self.costToGo[(newRow, newCol)] = new_cost_to_go
            self.path[(newRow, newCol)] = currentNode
            return True
        return False

    # animate path
    def animate(self, explored_states, backtrack_states, path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(path), fourcc, 20.0, (self.numCols, self.numRows))
        image = np.zeros((self.numRows, self.numCols, 3), dtype=np.uint8)
        count = 0
        for state in explored_states:
            image[self.numRows - state[0], state[1] - 1] = (255, 255, 0)
            if(count%80 == 0):
                out.write(image)
            count = count + 1

        count = 0
        for row in range(1, self.numRows + 1):
            for col in range(1, self.numCols + 1):
                if(image[self.numRows - row, col - 1, 0] == 0 and image[self.numRows - row, col - 1, 1] == 0 and image[self.numRows - row, col - 1, 2] == 0):
                    if(self.IsValid(row, col) and self.IsObstacle(row, col) == False):
                        image[self.numRows - row, col - 1] = (255, 255, 255)
                        if(count%80 == 0):
                            out.write(image)
                        count = count + 1
            
        if(len(backtrack_states) > 0):
            for state in backtrack_states:
                image[self.numRows - state[0], state[1] - 1] = (0, 0, 255)
                out.write(image)
                cv2.imshow('result', image)
                cv2.waitKey(5)
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        out.release()
    
    # diagonal heuristic
    def diagonal_heuristic(self, row, col):
        return max(np.abs(self.goal[0] - row) / self.stepSize, np.abs(self.goal[1] - col) / self.stepSize)

    # euc heuristic
    def euc_heuristic(self, row, col, weight = 1):
        return weight * np.sqrt((((self.goal[0] - row) / self.stepSize)**2) + (((self.goal[1] - col) / self.stepSize)**2))
    
    # a-star algo
    def search(self):
        # mark source node and create a queue
        exploredStates = []
        queue = []
        self.costToCome[self.start] = 0
        self.costToGo[self.start] = self.euc_heuristic(self.start[0], self.start[1])
        self.distance[self.start] = self.costToCome[self.start] + self.costToGo[self.start]
        heappush(queue, (self.distance[self.start], self.start))
        
        # run a-star
        while(len(queue) > 0):
            # get current node
            _, currentNode = heappop(queue)
            self.visited[currentNode] = True
            exploredStates.append(currentNode)
            
            # if goal node then break
            if(currentNode[0] == self.goal[0] and currentNode[1] == self.goal[1]):
                break
               
            # traverse the edges
            if(self.ActionMoveLeft(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][0], currentNode[0], currentNode[1] - self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0], currentNode[1] - self.stepSize)], (currentNode[0], currentNode[1] - self.stepSize)))
            
            if(self.ActionMoveRight(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][1], currentNode[0], currentNode[1] + self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0], currentNode[1] + self.stepSize)], (currentNode[0], currentNode[1] + self.stepSize)))
                    
            if(self.ActionMoveUp(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][2], currentNode[0] - self.stepSize, currentNode[1])
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] - self.stepSize, currentNode[1])], (currentNode[0] - self.stepSize, currentNode[1])))
                    
            if(self.ActionMoveDown(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][3], currentNode[0] + self.stepSize, currentNode[1])
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] + self.stepSize, currentNode[1])], (currentNode[0] + self.stepSize, currentNode[1])))
                    
            if(self.ActionMoveRightDown(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][4], currentNode[0] + self.stepSize, currentNode[1] + self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] + self.stepSize, currentNode[1] + self.stepSize)], (currentNode[0] + self.stepSize, currentNode[1] + self.stepSize)))
                    
            if(self.ActionMoveRightUp(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][5], currentNode[0] - self.stepSize, currentNode[1] + self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] - self.stepSize, currentNode[1] + self.stepSize)], (currentNode[0] - self.stepSize, currentNode[1] + self.stepSize)))
                    
            if(self.ActionMoveLeftUp(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][6], currentNode[0] - self.stepSize, currentNode[1] - self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] - self.stepSize, currentNode[1] - self.stepSize)], (currentNode[0] - self.stepSize, currentNode[1] - self.stepSize)))
                    
            if(self.ActionMoveLeftDown(currentNode[0], currentNode[1])):
                updateHeap = self.UpdateAction(currentNode, self.graph[currentNode][7], currentNode[0] + self.stepSize, currentNode[1] - self.stepSize)
                if(updateHeap):
                    heappush(queue, (self.distance[(currentNode[0] + self.stepSize, currentNode[1] - self.stepSize)], (currentNode[0] + self.stepSize, currentNode[1] - self.stepSize)))
                    
        # return if no optimal path
        if(self.distance[self.goal] == float('inf')):
            return (exploredStates, [], self.distance[self.goal])
        
        # backtrack path
        backtrackStates = []
        node = self.goal
        while(self.path[node] != -1):
            backtrackStates.append(node)
            node = self.path[node]
        backtrackStates.append(self.start)
        backtrackStates = list(reversed(backtrackStates))      
        return (exploredStates, backtrackStates, self.distance[self.goal])
