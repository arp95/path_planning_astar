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
from utils import *
import sys


startCol = int(input("Enter the x-coordinate for start node : "))
startRow = int(input("Enter the y-coordinate for start node : "))
goalCol = int(input("Enter the x-coordinate for goal node : "))
goalRow = int(input("Enter the y-coordinate for goal node : "))
radius = int(input("Enter the radius for the robot : "))
clearance = int(input("Enter the clearance for the robot : "))
stepSize = int(input("Enter the step size : "))

# take start and goal node as input
start = (startRow, startCol)
goal = (goalRow, goalCol)
astar = AStar(start, goal, clearance, radius, stepSize)

if(astar.IsValid(start[0], start[1])):
    if(astar.IsValid(goal[0], goal[1])):
        if(astar.IsObstacle(start[0],start[1]) == False):
            if(astar.IsObstacle(goal[0], goal[1]) == False):
                (exploredStates, backtrackStates, distanceFromStartToGoal) = astar.search()
                astar.animate(exploredStates, backtrackStates, "./astar_rigid.avi")

                # print optimal path found or not
                if(distanceFromStartToGoal == float('inf')):
                    print("\nNo optimal path found.")
                else:
                    print("\nOptimal path found. Distance is " + str(distanceFromStartToGoal))
            else:
                print("The entered goal node is an obstacle ")
                print("Please check README.md file for running astar.py file.")
        else:
            print("The entered start node is an obstacle ")
            print("Please check README.md file for running astar.py file.")
    else:
        print("The entered goal node outside the map ")
        print("Please check README.md file for running astar.py file.")
else:
    print("The entered start node is outside the map ")
    print("Please check README.md file for running astar.py file.")
