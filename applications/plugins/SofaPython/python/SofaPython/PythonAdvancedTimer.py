import os
import sys
import Sofa
# ploting
import matplotlib.pyplot as plt
# JSON deconding
from collections import OrderedDict
import json
# argument parser: usage via the command line
import argparse


def measureAnimationTime(node, timerName, timerInterval, timerOutputType, resultFileName, simulationDeltaTime, iterations):

    # timer
    Sofa.timerSetInterval(timerName, timerInterval) # Set the number of steps neded to compute the timer
    Sofa.timerSetEnabled(timerName, True)
    resultFileName = resultFileName + ".log"
    rootNode = node.getRoot()

    with open(resultFileName, "w+") as outputFile :
        outputFile.write("{")
        i = 0
        Sofa.timerSetOutputType(timerName, timerOutputType)
        while i < iterations:
            Sofa.timerBegin(timerName)
            rootNode.simulationStep(simulationDeltaTime)
            result = Sofa.timerEnd(timerName, rootNode)
            if result != None :
                outputFile.write(result + ",")
                oldResult = result
            i = i+1
        last_pose = outputFile.tell()
        outputFile.seek(last_pose - 1)
        outputFile.write("\n}")
        outputFile.seek(7)
        firstStep = outputFile.read(1)
        outputFile.close()
        Sofa.timerSetEnabled(timerName, 0)

        print "[Scene info]: end of simulation."
    return 0
