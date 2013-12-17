# importing this takes care of adding compliant python path to
# sys.path. you might want to symlink this file next to your actual
# scene files. this will become unnecessary in the future.

import Sofa
import sys

path = Sofa.src_dir() + '/applications/plugins/Compliant/python' 
# TODO avoid doing this twice ?
sys.path.append(path)



