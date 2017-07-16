The Python Scene Language for Sofa 
===========
The Python Scene Language (PSL) mixes the advantages of *XML* and *pyscn* in an unified and powerfull framework. 

PSL features:
- descriptive scenes (as XML)
- programable fragments (with embeded Python)
- scene templates (customisable dynamic element that can instantiated)
- libraries (for scene elements reuse and sharing)
- explicit aliasing (to simplify scene writing).
- preserve scene structure when it is loaded & saved.

To give you a taste of the language in its JSON flavor here is a small scene in which we import the SoftRobotActuator library. This library contains templates, on of them is the PneuNets actuator. Once imported, the template is then instanciated in the scene.  
```css
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

#### Introduction. 
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
```hjson
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

#### Templates
A Template is a component that stores a sub-graph in its textual, or parsed, form. The Template then can be instantiated 
in the graph.

```hjson
Node : {
	name : "root"
	Template : {
		name : "MyTemplate"
		properties : { name : "undefined"
			       numpoints : 3 
		}
		Node : {
			name : p"aName" 
			MechanicalObject: { position=p"range(0, numparts*3)" }
			UniformMass : {}
			Node : {
				name : "visual"
				BarycentricMapping : {}
				OglModel : { filename = "myOBJ.obj"}
			}
		}
	}

	/// The template can then be instantiated using its name as in:
	MyTemplate : {
		name : "defined1"
		numpoints : 100 
	}
	
	MyTemplate : {
		name : "defined2"
		numpoints : 10 
	}
	
	/// Or using Python 
	Python : '''
		for i in range(0,10):
			instantiate(root, "MyTemplate", {name:"defined"+str(i), numpoints : i})
		'''
}
```

#### Import 
To allow template re-usability it is possible to store them in file or directories that can be imported with the Import directive. 
In a file named mylibrary.pyjson" define  a template 
```hjson
	Template { name : "MotorA" ... }
	Template { name : "MotorB" ... }
	Template { name : "MotorC" .... }
```

Then in your scene file you load and use the template in the following way:
```hjson
Node : {
	Import : mylibrary 
	
	mylibrary.MotorA : {}
	mylibrary.MotorB : {}
	... 
}
```

##### Aliasing
In Sofa the aliasing system is implicit and the alias are defined in the sofa code source. This is really trouble some as users need to *discover* that in a scene "Mesh" is in fact an alias to a "MeshTopology" object. Without proper tools the solution is often to search in the source code which was an alias. 

In PSL we are preserving the use of Alias but we make them explicit. So each scene can defined its own set of alias and anyone reading the scene knows what are the alias and what are the real underlying objects. 
```hjson 
	Import : mylibrary 

	Alias : TSPhereModel-CollisionSphere
	Alias : mylibrary.MotorA-MotorA
	Alias : mylibrary.MotorB-MotorB
	
	/// Now we can use either
	TSPhereModel : {}
	
	/// or
	CollisionSphere : {}
```
