import Sofa

def createTree(node):
	A1 = node.createChild('A1')
	A2 = node.createChild('A2')
	B1 = A1.createChild('B1')
	B2 = A1.createChild('B2')
	B3 = A2.createChild('B3')
	B4 = A2.createChild('B4')