import sys
# ploting
import matplotlib.pyplot as plt
# JSON deconding
from collections import OrderedDict
import json
# argument parser: usage via the command line
import argparse

class TimerLjsonManyFilesPlot() :

    def __init__(self):
        lol = 0


    def parseInput(self):
        parser = argparse.ArgumentParser(
        description='Programm to plot light JSON file from a SOFA simulation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,# ArgumentDefaultsHelpFormatter
        epilog='''This program was made to create plot from a light JSON file from a SOFA simulation time capture. You can choose the componants and componants data to plot. It uses matplotlib (https://matplotlib.org/).''')
        parser.add_argument('componantName', metavar='j', type=str, help='Componant name to plot.')
        parser.add_argument('-v', type=str, default='Percent', help='Value to search on the capture values of each componant. valide values are : [Dev, Level, Max, Mean, Min, Num, Percent, Start, Total].')
        parser.add_argument('-j', nargs='+', help='JSON file path(s) of JSON file(s) to plot.')

        args = parser.parse_args()
        return parser,args;

    ###
     # Method : parseJsonComponantsId
     # Brief : parse a json file to create a block for the given composant name
     # Param : jsonData, json - data extracted from the json file
     # Param : componantID, string - id of the component to seek in the json file
     # Param : deep, int - 0 to get all componants on the same level than target, 1 to get all children of target
     ###
    def parseJsonComponantsId(self, jsonData, componantID, value) :

        # First iteration is used to create the list that will handle informations
        # The list is defiend as following :
        #     steps   | componantID |
        # 0  "Steps"  |  "CompName" |
        # 1     1     |    0.285    |
        #            ...

        parsedInformations = []
        firstPass = 1

        # Each k in this loop is the simulation step
        for k,v in jsonData.items() :

            # First analys to take Steps informations
            if firstPass == 1 :
                row = ["Steps", int(k)]
                parsedInformations.append(row)
                # Take informations from the target componant
                for kbis, vbis in v.items() :
                    if kbis == componantID :
                        row = []
                        row.append(componantID)
                        row.append(vbis["Values"][value])
                        parsedInformations.append(row)
                firstPass = 0

            # Informations extraction
            else :
                parsedInformations[0].append(int(k))
                for kbis, vbis in v.items() :
                    i = 0
                    if kbis == componantID :
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
    def parseJsonFile(self, componantID, value, *jsonFiles):
        jsonOpenedFiles = []
        jsonAnalysedFiles = []

        # Open and parse all files
        for jsonFile in jsonFiles :

            try :
                openedJsonFile = open(jsonFile, "r")
            except IOError:
                print "[ERROR]: The file " + jsonFile + " could not be opened."
                for openedFile in jsonOpenedFiles:
                    openedFile.close()
                return 0

            try:
                jsonParsedFile = json.load(openedJsonFile, object_pairs_hook=OrderedDict)
            except:
                print "[ERROR] Could not parse json file " + jsonFile
                continue

            jsonOpenedFiles.append(openedJsonFile)
            jsonAnalysedFiles.append(self.parseJsonComponantsId(jsonParsedFile, componantID, value))


        fig, ax = plt.subplots()
        lineColors = ["yellow", "green", "red", "blue", "orange", "black", "purple", "brown"]
        markStyles = ['.', '+', 'p', '*', 'o', 'v', '^', '<', '>', '8', 's', 'h', 'x', 'D', '2']
        lineColorIndice = 0
        markStyleIndice = 0
        fileNameIndice = 0

        # Ploting
        for fileInformation in jsonAnalysedFiles :
            for i in fileInformation :
                if i[0] != "Steps" :
                    labelName = jsonFiles[fileNameIndice] + "::" + i[0]
                    ax.plot(fileInformation[0][1:], i[1:], label=labelName, color=lineColors[lineColorIndice], marker=markStyles[markStyleIndice])
                    markStyleIndice = (markStyleIndice + 1) % len(markStyles)
                    fileNameIndice = fileNameIndice + 1
                lineColorIndice = (lineColorIndice + 1) % len(lineColors)

        # Create the legend of the plot
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')

        #legend.get_frame().set_facecolor('#00FFCC')
        plt.show()


        return 0

def main():
    # Create the object
    obj = TimerLjsonManyFilesPlot()

    # Parse the console input
    parser, args = obj.parseInput()
    componantID = args.componantName
    value = args.v
    jsonFiles = args.j

    obj.parseJsonFile(componantID, value, *jsonFiles)

    return

if __name__ == "__main__":
    main()
