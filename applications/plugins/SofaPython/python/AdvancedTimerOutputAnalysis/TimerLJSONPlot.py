import sys
# ploting
import matplotlib.pyplot as plt
# JSON deconding
from collections import OrderedDict
import json
# argument parser: usage via the command line
import argparse

class TimerLJSONPlot() :

    def __init__(self):
        lol = 0

    def parseInput(self):
        parser = argparse.ArgumentParser(
        description='Programm to plot light JSON file from a SOFA simulation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,# ArgumentDefaultsHelpFormatter
        epilog='''This program was made to create plot from a light JSON file from a SOFA simulation time capture. You can choose the componants and componants data to plot. It uses matplotlib (https://matplotlib.org/).''')
        parser.add_argument('jsonFileName', metavar='j', type=str, help='Filename of the JSON file to plot.')
        parser.add_argument('-d', type=int, default='0', help='Deepness of the analyse. 0 means an analyse of the comonant and the componants on the same level. 1 means an analyse of the componant and its children.')
        parser.add_argument('-v', type=str, default='Percent', help='Value to search on the capture values of each componant. valide values are : [Dev, Level, Max, Mean, Min, Num, Percent, Start, Total].')
        parser.add_argument('-c', nargs='+', default='Mechanical', help='Componant(s) to search on the JSON file.')

        args = parser.parse_args()
        return parser,args;


    ###
     # Method : parseJsonComponantsId
     # Brief : parse a json file to create a block for the given composant name
     # Param : jsonData, json - data extracted from the json file
     # Param : componantID, string - id of the component to seek in the json file
     # Param : deep, int - 0 to get all componants on the same level than target, 1 to get all children of target
     ###
    def parseJsonComponantsId(self, jsonData, componantID, deep, value) :
        parsedInformations = []

        # First iteration is used to create the list that will handle informations
        # The list is defiend as following :
        #     steps   | componantID | subComponant | subComponant2 | ...
        # 0  "Steps"  |  "CompName" | "subCompName"| "subCompName" | ...
        # 1     1     |    0.285    |      0.185   |      0.1      | ...
        #                        ...

        keyNumber = 0
        father = ""

        # Each k in this loop is the simulation step
        for k,v in jsonData.items() :

            # First analys to take search informations
            if keyNumber == 0 :
                row = ["Steps", k]
                parsedInformations.append(row)
                # Take informations from the target componant
                for kbis, vbis in v.items() :
                    if kbis == componantID :
                        if deep == 0 :
                            father = vbis["Father"]
                        else :
                            father = componantID
                        row = []
                        row.append(componantID)
                        row.append(vbis["Values"][value])
                        parsedInformations.append(row)
                # Take informations from componants on the same level than the target
                for kbis, vbis in v.items() :
                    if kbis != componantID and vbis["Father"] == father :
                        row = []
                        row.append(kbis)
                        row.append(vbis["Values"][value])
                        parsedInformations.append(row)
                keyNumber = 1

            # Informations extraction
            else :
                parsedInformations[0].append(int(k))
                for kbis, vbis in v.items() :
                    i = 0
                    if kbis == componantID or vbis["Father"] == father :
                        # Find the componant index in parsedInformations
                        for j, info in enumerate(parsedInformations) :
                            if info[0] == kbis :
                                i = j
                        # stock informations
                        row = parsedInformations[i]
                        row.append(vbis["Values"][value])

        return parsedInformations


    ###
     # Method : parseJsonFile
     # Brief : parse a json file to create a gnuplot file of the timer analysis
     # Param : jsonFile, string - name of the file to parse
     # Param : *componantsID, list of strings - ids of components to seek in the file
     ###
    def parseJsonFile(self, jsonFileName, deep, value, *componantsID):
        with open(jsonFileName, "r")  as jsonFile :
            jsonData = json.load(jsonFile, object_pairs_hook=OrderedDict)

            fig, ax = plt.subplots()
            lineColors = ["green", "red", "blue", "yellow", "orange", "black", "purple", "brown"]
            markStyles = ['.', '+', 'p', '*', 'o', 'v', '^', '<', '>', '8', 's', 'h', 'x', 'D', '2']
            lineColorIndice = 0
            markStyleIndice = 0

            for componantID in componantsID :
                result = self.parseJsonComponantsId(jsonData, componantID, deep, value)
                # Create plot
                for i in result :
                    if i[0] != "Steps" and i[0] != componantID:
                        ax.plot(result[0][1:], i[1:], label=i[0], color=lineColors[lineColorIndice], marker=markStyles[markStyleIndice])
                        markStyleIndice = (markStyleIndice + 1) % len(markStyles)
                    elif i[0] != "Steps" :
                        ax.plot(result[0][1:], i[1:], label=i[0], color=lineColors[lineColorIndice])
                legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
                # Set next line color
                lineColorIndice = (lineColorIndice + 1) % len(lineColors)

            legend.get_frame().set_facecolor('#00FFCC')
            plt.show()

            jsonFile.close()

        return 0

def main():
    # Create the object
    obj = TimerLJSONPlot()

    # Parse the console input
    parser, args = obj.parseInput()
    jsonFileName = args.jsonFileName
    deep = args.d
    value = args.v
    componantsID = args.c

    obj.parseJsonFile(jsonFileName, deep, value, *componantsID)

    return

if __name__ == "__main__":
    main()
