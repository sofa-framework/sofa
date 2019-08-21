#!/usr/bin/python
# coding: utf8 
#############################################################################
# This file is part of Sofa Framework
#
# This script is showing how you can use the launcher.py API to start
# multiple runSofa instance and gather the results. 
# 
# You need the cheetha template engine to use this
# http://www.cheetahtemplate.org/learn.html
#
# Contributors:
#       - damien.marchal@univ-lille.1
#############################################################################
import sys                 
from launcher import *                 
                       
filenames = ["example.scn","example.py"]
filesandtemplates = []
for filename in filenames:                
        filesandtemplates.append( (open(filename).read(), filename) )
        

################## EXAMPLE USING THE SEQUENTIAL LAUNCHER #################################
print("==================== NOW USING SEQUENTIAL LAUNCHER ===================")
results = startSofa([ {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                      {"GRAVITYXML": "9 2 4", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                      {"GRAVITYXML": "1 2 5", "GRAVITYPY": [1,2,1], "nbIterations":1000 } ], 
                    filesandtemplates, launcher=SerialLauncher())

for res in results:
       print("Results: ")
       print("    directory: "+res["directory"])
       print("        scene: "+res["scene"])
       print("      logfile: "+res["logfile"])
       print("     duration: "+str(res["duration"])+" sec")                 




################## EXAMPLE USING THE PARALLEL LAUNCHER #################################        
print("==================== NOW USING PARALLEL LAUNCHER =====================")
results = startSofa([ {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                    {"GRAVITYXML": "1 2 3", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                    {"GRAVITYXML": "1 2 3", "GRAVITYPY": [1,2,3], "nbIterations":1000 } ], 
                    filesandtemplates, launcher=ParallelLauncher(1))

for res in results:
       print("Results: ")
       print("    directory: "+res["directory"])
       print("        scene: "+res["scene"])
       print("      logfile: "+res["logfile"])
       print("     duration: "+str(res["duration"])+" sec")  




################## EXAMPLE USING THE DISTRIBUTED LAUNCHER #################################        
print("==================== NOW USING SSH LAUNCHER ===========================")                               
hosts=["192.168.0.22", "192.168.0.22"]            
results=startSofa([ {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                    {"GRAVITYXML": "9 2 4", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                    {"GRAVITYXML": "1 2 5", "GRAVITYPY": [1,2,1], "nbIterations":1000 },
                    {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                    {"GRAVITYXML": "9 2 4", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                    {"GRAVITYXML": "1 2 5", "GRAVITYPY": [1,2,1], "nbIterations":1000 },
                    {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                    {"GRAVITYXML": "9 2 4", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                    {"GRAVITYXML": "1 2 5", "GRAVITYPY": [1,2,1], "nbIterations":1000 },
                    {"GRAVITYXML": "0 0 0", "GRAVITYPY": [1,2,3], "nbIterations":1000 },  
                    {"GRAVITYXML": "9 2 4", "GRAVITYPY": [1,2,3], "nbIterations":1000 },
                    {"GRAVITYXML": "1 2 5", "GRAVITYPY": [1,2,1], "nbIterations":1000 } ], 
                    filesandtemplates, launcher=SSHLauncher(hosts, "YOULOGIN", 
                    runSofaAbsPath="THEPATHTOSOFA/runSofa"))
                    
# Start sofa returns a dictionnary 
for res in results:
       print("Results: ")
       print("    directory: "+res["directory"])
       print("        scene: "+res["scene"])
       print("      logfile: "+res["logfile"])
       print("     duration: "+str(res["duration"])+" sec")  
