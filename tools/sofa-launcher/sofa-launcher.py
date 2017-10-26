#!/usr/bin/python
# coding: utf8 
#############################################################################
# This file is part of Sofa Framework
#
# Need Cheetah template engin to work (http://www.cheetahtemplate.org/learn.html) 
# 
# Contributors:
#       - damien.marchal@univ-lille.1
#############################################################################
import sys             
import json    
import time
import datetime 
from launcher import *                 

if __name__ == '__main__':
        if len(sys.argv) != 2:
                print("USAGE: sofa-launcher.py youfile.json")
                sys.exit(-1)

        cfgfilename = sys.argv[1]
        cfgfile = open(sys.argv[1]).read()
        cfg = json.loads(cfgfile)
        
        starttime = datetime.datetime.now() 
        begintime = time.time() 
        
        # Create the adequate launcher
        launcher = SerialLauncher() 
        if cfg["launcher"] == "sequential":
                pass                 
        elif cfg["launcher"] == "parallel":
                launcher = ParallelLauncher( cfg["numjobs"] )
        elif cfg["launcher"] == "ssh":
                launcher = SSHLauncher(cfg["sshhosts"], cfg["sshlogin"], cfg["sshsofapath"]) 
        else:
                print("Missing parser in the configuration file")
                sys.exit(-1)
                
                
        # Prepare the files and the templates for processing. 
        filesandtemplates = []
        for filename in cfg["files"]:                
                filesandtemplates.append( (open(filename).read(), filename) )
        
        
        # Start the jobs... and print the results         
        results = startSofa( cfg["variables"], filesandtemplates, launcher)
        
        endtime = time.time() 
        
        template = open(cfg["resultsstyle"]).read()
        theFile = open(cfg["resultsfile"], "w+") 
        t = Template(template, searchList={"results" : results, 
                                           "starttime" : str(starttime), 
                                           "duration" : endtime-begintime, 
                                           "cfgfilename" : cfgfilename })
        theFile.write(str(t))
        
