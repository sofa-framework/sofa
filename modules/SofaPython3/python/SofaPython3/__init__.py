print("LOADING PYTHON3")

import test
print("TEST")

#import .RunTime
from .RunTime import load

from .SceneLoaderPY3 import registerLoader
registerLoader()

from .Base import *
print(str(dir(Base)))
