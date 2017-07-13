The Python Scene Language for Sofa 
===========
The Python Scene Language mix advantage of XML and PyScn in a single and powerful framework. 
It features:
- structural and descriptive scenes (as XML scene)
- programable and dynamic fragment (with embeded Python)
- scene templates (customisable dynamic element that can be reused and instantiated)
- templates libraries (for scene element reuse and sharing)
- explicit aliasing (to simplify scene writing)
And it can be loaded & saved in a consistant way. 

To give you a taste of the language here is a small scene that import a library of scene element (here our SoftRobotActuator library) and instantiate a PneuNets actuator from it. It also create several dynamic node...just to show the syntax: 

```hjson
Node {
	name : "myNameIsRoot"

	Import : SoftRobotActuators
	Alias : SoftRobotActuators.PneuNets-PneuNets

	Node : {
		Python : {
			Sofa.msg_info(root, "PSL offer scene direct scene element access to python code with scoping !!!")
			for i in range(0,10):
				self.addChild("one")
				parent.addChild("two")
				myNameIsRoot.addChild("three")
		}
	}

	PneuNets : { 
		name:"myPneuNet" 
		numSections : 10
	}
}
```

We hope this example gave you some envy to learn more about it. Let's start with a big longer description of the langage feature and syntax. The language itself can be defined in term of abstract syntax or through a given concrete syntax. For the simplicity of the following we will employ the H-JSON concrete syntax as it provides both readbility, compactness and clarity. This H-JSON is currently implemented in Sofa but please keep in mind that other alternatives are possible based on XML or YAML instead of H-JSON. 

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
```hjson
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




