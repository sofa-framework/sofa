#!/usr/bin/python
# coding: utf8 
#############################################################################
# A small utility to launch multiple sofa in parallel.
#
# Need Cheetah template to work. 
# 
# Contributors:
#       - damien.marchal@univ-lille.1
#############################################################################
import threading
import Queue
import tempfile 
import sys
from Cheetah.Template import Template
import tempfile
import shutil
import re
import time

from subprocess import Popen, PIPE, call


class Launcher(object):
        def start(tasks):
                pass

class SerialLauncher(Launcher):
        def start(self, tasks):
                results=[]
                for (numiterations, directory, scene, log)  in tasks:
                        print("[0] sequential new sofa task in: " + str(scene))
                        begin = time.time()
                        try:
                                a = Popen(["runSofa", "-g", "batch", "-l", "SofaPython", "-n", str(numiterations), scene], cwd=directory, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                        except:
                                print("Unable to find runSofa, please add the runSofa location to your PATH and restart sofa-launcher.")
                                sys.exit(-1)
                        astdout, astderr = a.communicate()
                        a.stdout.close()
                        a.stderr.close() 
                        end = time.time()
                        logfile = open(log, "w+")
                        logfile.write("========= STDOUT-LOG============\n")
                        logfile.write(astdout)
                        logfile.write("========= STDERR-LOG============\n")
                        logfile.write(astderr) 
                        logfile.close()

                        results.append({
                                "directory" : directory,
                                "scene" : scene,
                                "logfile" : log,
                                "logtxt"  : open(log).read(),
                                "duration" : end-begin 
                                } )
                return results 
                
class ParallelLauncher(Launcher):
        def __init__(self, numprocess):
                self.numprocess = numprocess
                self.pendingtask = Queue.Queue()
                self.times = {}
                               
                # Create the threads
                for i in range(0, numprocess):
                        t = threading.Thread(target=ParallelLauncher.worker, args=[self])
                        t.daemon = True
                        t.start()
        
        def worker(self):
                while True:
                        task = self.pendingtask.get() 
                        (numiterations, directory, scene, log) = task 
                        print("[{0}] processing threaded sofa task in: {1}".format(threading.currentThread().ident, scene))
                        
                        begin = time.time()
                        try:
                                a = Popen(["runSofa", "-l", "SofaPython", "-g", "batch", "-n", str(numiterations), scene], cwd=directory, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                        except:
                                print("Unable to find runSofa, please add the runSofa location to your PATH and restart sofa-launcher.")
                                sys.exit(-1)

                        astdout, astderr = a.communicate()
                        a.stdout.close()
                        a.stderr.close() 
                        end = time.time()
                        
                        self.times[log] = end-begin 
                        logfile = open(log, "w+")
                        logfile.write("========= STDOUT-LOG============\n")
                        logfile.write(astdout)
                        logfile.write("========= STDERR-LOG============\n")
                        logfile.write(astderr) 
                        logfile.close()

                        #logfile.write("========== MATCH-LOG ===========\n")
                        #logfile.write(str(filtering(astdout)))                        
                                                
                        self.pendingtask.task_done()
                        
                                
        def start(self, tasks):
                for task in tasks:
                        self.pendingtask.put(task) 
        
                self.pendingtask.join()
                
                results=[]
                for task in tasks:
                        (numiterations, directory, scene, log) = task 
                 
                        results.append({
                                "directory" : directory,
                                "scene" : scene,
                                "logfile" : log,
                                "logtxt"  : open(log).read(),
                                "duration" : self.times[log] 
                                } )        
                                
                return results

class SSHLauncher(Launcher):
        def __init__(self, hosts, login, runSofaAbsPath="runSofa"):
                self.login = login
                self.hosts = hosts
                self.runSofa = runSofaAbsPath
                self.pendingtask = Queue.Queue()
                self.times = {} 
                                               
                # Create the threads
                for host in hosts:
                        t = threading.Thread(target=SSHLauncher.worker, args=[self, host])
                        t.daemon = True
                        t.start()
        
        def worker(self, host):
                print("Thread created for host: "+self.login+"@"+host)
                while True:
                        task = self.pendingtask.get() 
                        (numiterations, directory, scene, log) = task 
                        sofacmd = "{2} -g batch -l SofaPython -n {0} {1}".format(numiterations, scene, self.runSofa)
                        print("[{0}] processing ssh sofa task: {1}".format(threading.currentThread().ident, sofacmd)) 
                        begin = time.time()
                        ssh = Popen(["ssh", "%s" % host, sofacmd], shell=False, stdout=PIPE, stderr=PIPE)
                        
                        astdout, astderr = ssh.communicate()
                        ssh.stdout.close()
                        ssh.stderr.close() 
                        
                        end = time.time()
                        self.times[log] = end-begin 
                        
                        logfile = open(log, "w+")
                        logfile.write("========== "+host+"==========\n")
                        logfile.write("========= STDOUT-LOG============\n")
                        logfile.write(astdout)
                        logfile.write("========= STDERR-LOG============\n")
                        logfile.write(astderr) 
                        logfile.write("========== MATCH-LOG ===========\n")
                        logfile.close()
                                                
                        self.pendingtask.task_done()
                        
                                
        def start(self, tasks):
                for task in tasks:
                        self.pendingtask.put(task) 
        
                self.pendingtask.join()
                results=[]
                for task in tasks:
                        (numiterations, directory, scene, log) = task 
                 
                        results.append({
                                "directory" : directory,
                                "scene" : scene,
                                "logfile" : log,
                                "logtxt"  : open(log).read(),
                                "duration" : self.times[log] 
                                } )        
                                
                return results


def startSofa(parameters, filesandtemplates, launcher):
        tasks = []
        
        if not isinstance(parameters, list):
                raise TypeError("the first parameter must be an list like structure of dictionnaries")

        if not isinstance(parameters, list):
                raise TypeError("parameters must be a list like structure of tuple")

        for param in parameters:
                tempdir = getTemporaryDirectory() 
                
                files=[]
                for (template,filename) in filesandtemplates:
                        files.append( tempdir+"/"+filename )
                        param["FILE"+str(len(files)-1)] = files[-1]
                        
                i=0        
                for (template,filename) in filesandtemplates:
                        theFile = open(files[i], "w+") 
                        t = Template(template, searchList=param)
                        theFile.write(str(t))
                        theFile.close()
                        i+=1
                        
                tasks.append((param["nbIterations"], tempdir, param["FILE0"], tempdir+"/output.log")) 
        
        return launcher.start(tasks)                        
        

def getTemporaryDirectory():
        return tempfile.mkdtemp(prefix="sofa-launcher-")

