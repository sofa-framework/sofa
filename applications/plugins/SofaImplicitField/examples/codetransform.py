import pyximport; pyximport.install()
import powered

import math

class Length(object):
	def __init__(self, t):
		self.v = t

	def compile(self):
		p = self.v.compile()		
		a=["mul", p[0], p[0]]
		b=["mul", p[1], p[1]]
		c=["mul", p[2], p[2]]
		d=["add", a, b]
		e=["add", d, c]
		f=["sqrt", e]
		return [f]
		
	def __sub__(self, right):
		return Sub(self, right)

class Sub(object):
	def __init__(self, a, b):
		self.a = a 
		self.b = b
		
	def compile(self):
		left = self.a.compile()
		right = self.b.compile()
		print("COMPILE: "+str(left) + " <with> " + str(right) + " " + str(type(self.b))) 
		if len(left) == 1 and len(right) == 1:
			a = ["sub", left, right]
			return [a]
		if len(left) == 3 and len(right) == 3:
			a = ["sub", left[0], right[0]]
			b = ["sub", left[1], right[1]]
			c = ["sub", left[2], right[2]]
			return [a,b,c]
		if len(left) == 3 and len(right) == 1:
			a = ["sub", left[0], right]
			b = ["sub", left[1], right]
			c = ["sub", left[2], right]
			return [a,b,c]
		
		print("Missing '-' operator with " +str(type(left)) +" and " + str(type(right)))
		
class Vector(object):
	def __init__(self, v):
		self.v=[Constant(v[0]),Constant(v[1]),Constant(v[2])]

	def compile(self):
		return [self.v[0].compile(), self.v[1].compile(), self.v[2].compile()]
		 
	def __getitem__(self,i):
		return self.v[i]	

	def __sub__(self, v):
		return Sub(self, v)		

class Constant(object):
	def __init__(self, v):
		self.v = v
		
	def compile(self):
		return [self.v]

class Memory(object):
	def __ini__(self, v):
		self.v = v
		
	def compile(self):
		return ('read &', self.v)		


class Sphere(object):
	def __init__(self, position=(1.0,0.0,0.0), radius=0.5):
		self.position = Vector(position)
		self.radius = Constant(radius)
	
	def compile(self, inp=Vector(("pos.x()","pos.y()","pos.z()"))):
		d = Length(self.position-inp) - self.radius
		return d.compile()

class Difference(object):
	def __init__(self, left, right):
		self.left = left
		self.right = right
		
	def compile(self, inp=Vector(("pos.x()","pos.y()","pos.z()"))):
		left = self.left.compile(inp)
		right = self.right.compile(inp)
		a=["sub", Constant(0.0).compile(), left]
		b=["max", a, right]
		return b

ssa=[]
code=""
def dumpInCython(t):
	global ssa
	print("SIZE OF T"+str(len(t)))
	if len(t) == 3:
		op, left, right = t
		r1 = dumpInCython(left)
		r2= dumpInCython(right)		
		ssa.append((len(ssa), [op, r1, r2]))
		return len(ssa)-1
	if len(t) == 2:
		op, left = t
		r = dumpInCython(left)
		ssa.append((len(ssa), [op, r]))
		return len(ssa)-1		
	if len(t) == 1:
		if isinstance(t[0], list):
			return dumpInCython(t[0])
		else:
			ssa.append((len(ssa), t))
			return len(ssa)-1		

def constantFolding(ssa):
	cstmap = {}
	for i in range(0,len(ssa)):
		k,v = ssa[i]
		if str(v) in cstmap:
			cstmap[str(v)].append(k)
		else:
			cstmap[str(v)]=[k]

	for k in cstmap:
		o = cstmap[k][0]
		nssa = ssa
			
		for id in cstmap[k][1:]:
			print("Replacing "+str(id)+" by "+str(o))
			nssa = []
			for k,v in ssa:
			
				if len(v) == 2 and v[1] == id:
					nssa.append( (k,[v[0],o]) )
				elif len(v) == 3 and v[2] == id:
					nssa.append( (k,[ v[0],v[1],o]) )
				elif len(v) == 3 and v[1] == id:	
					nssa.append( (k,[ v[0],o, v[2]]) )
				elif k != id:
					nssa.append((k , v)) 
					
			ssa=nssa
	return nssa	 
expr = Difference( Sphere((0.5,0.5,0.5)), Sphere((0.1,0.5,0.5)) )
import pprint
pprint.pprint(expr.compile())

dumpInCython(expr.compile())
pprint.pprint(ssa)
l=-1
while l != len(ssa):
	l = len(ssa) 
	ssa=constantFolding(ssa)
pprint.pprint(ssa)

def dumpInC(ssa):
	r=""
	for k, v in ssa:
		r += "double ssa"+str(k)+" = " ;
		if len(v) == 1:
			r+= str(v[0])
		elif len(v) == 2:
			r+= str(v[0])+"(ssa"+str(v[1])+")" 
		elif len(v) == 3:
			if v[0] == "sub":
				r+= "ssa"+str(v[1])+" - ssa"+str(v[2])+"" 
			elif v[0] == "add":
				r+= "ssa"+str(v[1])+" + ssa"+str(v[2])+"" 
			elif v[0] == "mul":
				r+= "ssa"+str(v[1])+" * ssa"+str(v[2])+"" 
			else:
				r+= str(v[0])+"(ssa"+str(v[1])+", ssa"+str(v[2])+")" 
	
		r += ";\n";
	return r	
print("CODE: \n"+dumpInC(ssa))

