#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import subprocess
import sys
import os
import time

# Brief:
# A simple script to load and run one time step of each scene of the examples repository 

scenespaths = [os.path.dirname(os.path.abspath(__file__))+'/../stlib/',
               os.path.dirname(os.path.abspath(__file__))+'/../splib/']

sofabin=""
if len(sys.argv) > 1: sofabin  = sys.argv[1]
if not os.path.isfile(sofabin): sofabin = os.path.dirname(os.path.abspath(__file__))+'/../../../../build-release/bin/runSofa'
if not os.path.isfile(sofabin): sofabin = os.path.dirname(os.path.abspath(__file__))+'/../../../../build/bin/runSofa'
if not os.path.isfile(sofabin): sofabin = os.path.dirname(os.path.abspath(__file__))+'/../../../../../build-release/bin/runSofa'
if not os.path.isfile(sofabin): sofabin = os.path.dirname(os.path.abspath(__file__))+'/../../../../../build/bin/runSofa'

if os.path.isfile(sofabin): 

      supportedExtensions = [".pyscn", ".scn", ".psl", ".py"]
      pattern = "/"

      if len(sys.argv) > 2:
	      pattern = sys.argv[2]

      #Global variables 
      global nbFailedTest
      global failedTestPathList
      global nbSucceedTest
      global totalTime
      nbFailedTest = 0
      nbSucceedTest = 0
      failedTestPathList = []
      totalTime = 0

      outputFilename = os.path.dirname(os.path.abspath(__file__))+"/TestExampleScenesOutput.txt"	      
      if os.path.isfile(outputFilename): 
        os.remove(outputFilename)
	         
      def dotest(arg, dirname, names):

	      global nbFailedTest
	      global failedTestPathList
	      global nbSucceedTest
	      global totalTime

	      GREEN = "\033[1;32m"
	      RED   = "\033[1;31m"
	      ENDL  = "\033[0m"

	      outputFile = open(outputFilename,"a")

	      for thisfile in names:

		      filename = os.path.join(dirname, thisfile)
		      b,ext = os.path.splitext(filename)
		      if os.path.isfile( filename ) and ext in supportedExtensions and pattern in filename:
			      print("[TESTING] "+dirname+"/"+thisfile, end=""),
			      sys.stdout.flush()
			      outputFile.write("==================== Testing: "+dirname+"/"+thisfile+"=========================== \n")
			      start_time = time.time()
			      retcode = subprocess.call([sofabin, filename, "-a" ,"-g","batch", "-l", "SofaPython", "-n", "1"], stdin=None, stdout=outputFile, stderr=outputFile)
			      testTime = (time.time() - start_time)

			      if retcode == 0 :
				      nbSucceedTest+=1
				      print(" (", end="")
				      print(float("{0:.2f}".format(testTime)), end="")
				      print("s)", GREEN+"[SUCCEED]"+ENDL)
				      totalTime += float("{0:.2f}".format(testTime))
			      else:
				      nbFailedTest+=1
				      failedTestPathList.append(filename)
				      print(" (",float("{0:.2f}".format(testTime)),"s)", RED+"[FAILED]"+ENDL)
				      totalTime += float("{0:.2f}".format(testTime))


	      outputFile.close()
	      
      for path in scenespaths:	      
              os.path.walk(path, dotest, None) 

      print("====================== [SUMMARY] ========================")
      print("Test: Load and run one time step of each scene of the stlib repository.")
      print("Total time: ",totalTime,"s")
      print("Succeed test(s): "+str(nbSucceedTest))
      if nbFailedTest == 0 : print("Failed  test(s): "+str(nbFailedTest))
      else : print("Failed  test(s): "+str(nbFailedTest), file=sys.stderr)
      for thisfile in failedTestPathList:
	      print("\t -"+thisfile, file=sys.stderr)
	      
else:
  
  print("TestExamplesScenes requires the path to runSofa.\n"
	"1- You can either give the explicit path by argument\n"
	"2- Or change the path directly in the python file")
