import os
import sys
sys.path.append(os.path.abspath('../python'))

from SofaPython import reloadhack

frame = reloadhack.ImportFrame() 
for i in range(0,10):
	print("After this line I should print only one 'I'm RELOADED' (not two).")
	frame = reloadhack.ImportFrame() 	
	import AFileToReload
	import AFileToReload
	frame.uninstall()
