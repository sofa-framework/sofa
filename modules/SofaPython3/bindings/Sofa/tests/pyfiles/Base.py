import Sofa
import array
import timeit
import unittest

def test_DataDict(o):
        print("Test data container")

        print(o.gravity)
        print(o.__data__["gravity"])
        
        print("SIZE:" +str(len(o.__data__))+"...")
        
        for c in list(o.__data__.items()):
            print(c)
        

def createScene(root):
        #test_DataDict(root)
        print("COucou")
        
