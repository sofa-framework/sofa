The PSL language for Sofa 
===========

PSL is a new scene description langage for the sofa framework. 

To give you a taste of the language here is a small scene that import a library of scene element and instantiate 
a PneuNets actuator from it.

```json
Node {
	name : "root"

	Import : SoftRobotActuators
	Alias : SoftRobotActuators.PneuNets-PneuNets

	Python : {
		Sofa.msg_info(root, "PSL is very cool")
	}

	PneuNets : { 
		name:"myPneuNet" 
		numSections : 10
	}
}
```

The rationals underlying the language design are to combine 
the advantage of descriptive language (as is XML) as well as the advantages of dynamicity through procedural programming. 
To do so the language is merging concept from both SCN files and PYSCN in a single unified framework. 
Some good features of the languages:
- stuctural and descriptive (as the XML the language structure match the scene)
- programmable with python
- support template (a group of pre-defined dynamic element that can be reused and instantiated)
- support importing libraries of template (for scene element reuse and sharing)
- support explicit aliasing (to simplify scene writing)

The language can be defined in term of abstract syntax and using a concrete syntaxe. For the simplicity of the following 
we will employ the H-JSON concrete syntax as it provides both readbility, compactness and clarity. This H-JSON is currently 
implemented but other alternative are also possible eg. XML or YAML ones. 

Let's start with a simple scene example in XML
```xml
<Node name="root">
	<Node name="child1">
		<MechanicalObject name="mstate"/> 
		<OglModel filename="anObj.obj"/> 
	</Node>
</Node>
}
```

The equivalent scene PSL(HJSON) is the following 
```json
Node {
	name : "root"
	Node {
		name : "child1"
		MechanicalObject: { name : "mstate" }
		OglModel : { filename : "anObj.obj" }
	}
}
```

The drawback of XML is that everything is static. This is why more and more people are using python to describe scene as it allow
 to write things like that: 
```python
root = Sofa.createNode("root")
child1 = root.createNode("child1")
child1.createObject("MechanicalObject", name="mstate")
child1.createObject("OglModel", name="anObj.obj") 
for i in range(0,10):
	child1.createNode("child_"+str(i))
```

The equivalent scene PSL(HJSON) is the following 
```json
Node {
	name : "root"
	Node {
		name : "child1"
		MechanicalObject: { name : "mstate" }
		OglModel : { filename : "anObj.obj" }
		Python : '''
			for i in range(0, 10):
				child1.createNode("child_"+str(i))
		'''
	}
}
```

At first sight the PSL version look a bit more complex. But it solve a deep problem of the python version. It can be loaded & saved and preserving the scene structure ! 
This is because in python the script is executed (consumed) at loading time and is not part of the scene so the only possible saving is to 
store the *result* of the execution of the script as in: 
```python
root = Sofa.createNode("root")
child1 = root.createNode("child1")
child1.createObject("MechanicalObject", name="mstate")
child1.createObject("OglModel", name="anObj.obj") 
child1.createNode("child_0")
child1.createNode("child_1")
child1.createNode("child_2")
child1.createNode("child_3")
child1.createNode("child_4")
child1.createNode("child_5")
child1.createNode("child_6")
child1.createNode("child_7")
child1.createNode("child_8")
child1.createNode("child_9")
```

With PSL, this does not happen because dynamic fragment of the language are stored un-executed in the scene graph. They can thus be easily saved in their initial form. 




