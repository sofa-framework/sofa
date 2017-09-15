from os import system, listdir, path, walk
system("clear")
LOGFILE = open("LogRSSFF.txt", "w")
desktop_dir = r"../"
#desktop_dir = r"/home/rtrivi/wor/Sofa/src/sofa-stable_v16.08"

for root, dirs, files in walk(desktop_dir):
	for file in files:
		with open(path.join(root, file), "r") as auto:
			for i, line in enumerate(auto, 0):# for line in auto:
				for word in line.split():
			  		if word in {"<RestShapeSpringsForceField"}:
						LOGFILE.write("<RestShapeSpringsForcefield found in " + file + " {" + str(i) + "}\n")
						for word in line.lower().split():
							if(word.find("external_rest_shape=\"") > -1 and word.find("@") == -1):
								with open(path.join(root, file), "r") as filer:
									s = filer.readlines()
									LOGFILE.write("\tLINE bef: " + str(s[i]))
									s[i] = s[i].replace("external_rest_shape=\"", "external_rest_shape=\"@")
									LOGFILE.write("\tLINE aft: " + str(s[i]))
								with open(path.join(root, file), "w") as filew:
									filew.writelines(s)
LOGFILE.close()