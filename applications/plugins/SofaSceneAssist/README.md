The Python Scene Language for Sofa 
===========
The Python Scene Language (PSL) mixes the advantages of *XML* and *pyscn* in an unified and powerfull framework. 
PSL features:
- descriptive scenes (as XML)
- programable fragments (with embeded Python)
- scene templates (customisable dynamic element that can instantiated)
- libraries (for scene elements reuse and sharing)
- explicit aliasing (to simplify scene writing).

And, compared to Python, it preserve scene structure when it is loaded & saved.

To give you a taste of the language in its JSON flavor here is a small scene in which we import the SoftRobotActuator library. This library contains templates, on of them is the PneuNets actuator. Once imported, the template is then instanciated in the scene.  
```hjson
Node {
	name : "myNameIsRoot"

	Import : SoftRobotActuators
	Alias : SoftRobotActuators.PneuNets-PneuNets

	Node : {
		Python : ''''
			Sofa.msg_info(root, "PSL offer scene direct scene element access to python code with scoping !!!")
			for i in range(0,10):
				self.addChild("one")
				parent.addChild("two")
				myNameIsRoot.addChild("three")
		'''
	}

	PneuNets : { 
		name:"myPneuNet" 
		numSections : 10
	}
}
```

We hope this example gave you some envy to learn more about it. Let's start with a big longer description. 

#### The PSL language. 
The language itself is defined either in term of abstract syntax or through a given concrete syntax. For the simplicity of the following we will employ the H-JSON concrete syntax as it provides both readbility, compactness and clarity. This H-JSON flavor of the language is currently implemented in Sofa but keep in mind that other alternatives are possible based on XML or YAML instead of H-JSON. 

Let's start with a simple scene example in XML
```xml
<Node name="root">
	<Node name="child1">
		<MechanicalObject name="mstate"/> 
		<OglModel filename="anObj.obj"/> 
	</Node>
</Node>
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

The drawback SCN files is that everything is static. This is why more and more people are using python 
to describe scene as it allows to write: 
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

At first sight the PSL version look a bit more complex. But it solve a deep problem of the python version. It can  preserve the scene structure when it is loaded & saved. This is because in python scenes the script is executed (consumed) at loading time and is not part of the scene. The consequence is that the only possible saving is to store the *result* of the execution of the script, totally loosing the advantages of python as visible in the previous scene saved in python: 
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

With PSL, this is not a problem because the dynamic fragment are stored *un-executed* in the scene graph. They can thus be easily modifie, re-run and saved. 
