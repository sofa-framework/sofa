import pyximport; pyximport.install()
import powered

import math

class Sphere(object):
	def __init__(self, position=[0,0,0], radius=0.5):
		self.position = position
		self.radius = radius

	def eval(self, x,y,z):
		x = x - self.position[0]
		y = y - self.position[1]
		z = z - self.position[2]

		return math.sqrt( (x*x + y*y + z*z) ) -self.radius

class Difference(object):
	def __init__(self, a, b):
		self.left = a 
		self.right = b

	def eval(self, x,y,z):
		leftv = self.left.eval(x,y,z)
		rightv = self.right.eval(x,y,z)
		return max(leftv, -rightv) 		


def evalField(x,y,z):
	f=Difference( Sphere([0.5, 0.5, 0.5], 0.2),
				  Sphere([0.3, 0.5, 0.5], 0.2) ) 
	f=Sphere([0.5,0.5,0.5], 0.3)
	return f.eval(x,y,z)
