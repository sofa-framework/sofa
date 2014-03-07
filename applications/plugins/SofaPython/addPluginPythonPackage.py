import os
import os.path
import sys
import Sofa

# python modules in plugins
path=os.path.join(Sofa.src_dir(),'applications/plugins')
pluginPathList = [os.path.join(path, p) for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
path=os.path.join(Sofa.src_dir(),'applications-dev/plugins')
pluginPathList += [os.path.join(path, p) for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
for pluginPath in pluginPathList :
#	print pluginPath, os.path.join(pluginPath,'python')
	if os.path.exists(os.path.join(pluginPath,'python')):
		print "Added python package in", os.path.join(pluginPath,'python')
		sys.path.append(os.path.join(pluginPath,'python'))
