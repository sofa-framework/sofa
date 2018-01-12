Python Scene Language for Sofa
===========
The Python Scene Language (PSL) project is an attempt to mix the advantages of *scn* and *pyscn* in an
unified framework. PSL provides an abstract language that has been specifically designed to make sofa
scenes in an elegant and powerful way. Compared to classical *.scn*, PSL offer scene dynamicity,
Compared to *.pyscn*, PSL offer savability, templates and a more compact declarative syntax

Index:
- [Installation](#installation-and-requirement)
- [Introduction](#introduction)
    - [First examples](#first-examples)
        - [Example with the XML syntax](#writing-scene-with-the-xml-syntax)
        - [Example with the HJSON syntax](#writing-scene-with-the-h-json-syntax)
    - [Adding python fragments](#python-fragments)
    - [Templates](#templates)
    - [Templates libraries](#import)
    - [Custom alias](#aliasing)
    - [Properties](#properties)
- [Advanced features](#advanced-features)
    - [Dynamic templates & GUI (DOC TODO)](#dynamic-templates-&-gui-interaction)
    - [Python DSL](#pure-python-dsl)
    - [Writing templates in python (DOC TODO)](#writing-templates-in-pure-python)
    - [Extending PSL](#extending-psl)
    - [PSLX](#writing-psl-scene-with-xml-syntax)


# Installation and requirement.
PSL can propose several alternatives syntaxes. Files ending with *.psl* are referring to H-JSON, a syntax
that look like QML or JSON. While files ending with *.pslx* use the XML syntax. The XML one is provided by
default, you want to use the H-JSON syntax you need to install H-JSON parser that is available at: http://hjson.org/
To do that you can do :
```shell
git clone https://github.com/hjson/hjson-py.git
cd hjson-py
sudo python setup.py install
```

It is also a good idea to install the syntax coloring scheme to handle *.psl* files in Kate or qtcreator.
The color scheme is provided in the file [psl-highlighting.xml](./psl-highlighting.xml). Installation
instructions for qtcreator are given [here](http://doc.qt.io/qtcreator/creator-highlighting.html).

# Introduction.
PSL features:
- a declarative scene language (as is .scn) that can be loaded & saved.
- with multiple alternative syntax (xml, hjson, python-pickled)
- with procedural elements (embeded Python)
- with scene templates (dynamic element that can be reuse and instantiated)
- with scene libraries (to store scene templates for reuse and sharing)
- with explicit aliasing (to simplify scene writing).
- with dedicated python DSL to simpliyfy scene writing in pure-python
- ... more to come ...

For the simplicity of the following we will employ the so called *H-JSON syntax* as it provides both readbility,
compactness and clarity.

## First example

### Writing scene with the H-JSON syntax.
Let's start with a very simple sofa scene that only contains two hierarchical nodes,
and OglModel and a MechanicalObject. Using the XML scene syntax this would be written
this way: 
```xml
<Node name="root">
        <Node name="child1">
                <MechanicalObject name="mstate"/>
                <OglModel filename="anObj.obj"/>
        </Node>
</Node>
```

Using PSL/HJSON syntax, the same scene would be written like that:
```css
Node : {
        name : "root"
        Node : {
                name : "child1"
                MechanicalObject: { name : "mstate" }
                OglModel : { filename : "anObj.obj" }
        }
}
```

The problem of our classic xml *.scn* files is that everything is static. For this reason ones often use
python to describe scenes as it allows to write:
```python
root = Sofa.createNode("root")
child1 = root.createNode("child1")
child1.createObject("MechanicalObject", name="mstate")
child1.createObject("OglModel", name="anObj.obj")
for i in range(0,10):
        child1.createNode("child_"+str(i))
```
The drawback of doing this, in addition to the poor visual emphasizing of the Sofa component lost in
the middle of python code is that once loaded this good looking python scene is in fact serialized as this one:
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

This is because in *.pyscn* the python code is consumed at loading time to generate the scene.
The consequence is that saving the scene is in fact storing the *result* of the execution of the script
and thus we are totally loosing the advantages of python.


With PSL it is possible to implement the following dynamic behavior in the following way.
```css
Node : {
        name : "root"
        Node : {
                name : "child1"
                MechanicalObject: { name : "mstate" }
                OglModel : { filename : "anObj.obj" }
        }

        Python : ''''
                  Sofa.msg_info(myNameIsRoot, "Hello world")
                  for i in range(0,10):
                        myNameIsRoot.addChild("three")
                  '''
}
```

The advantage of PSL is that the python fragment are stored into the scene in their textual form
and can thus be saved or re-executed.

### Python fragments
In PSL it is possible to add python code to your scene using the *Python* component as in:
```css
Node : {
        name : "root"
        Node : {
                name : "child1"
                MechanicalObject: { name : "mstate" }
                OglModel : { filename : "anObj.obj" }
                Python : '''
                         for i in range(0, 10):
                                child1.createChild("child_"+str(i))
                         '''
        }
}
```

To simplify scene writing the scenes elements ("root, child1", ...) within the scene graph "scope"
are exposed in the Python component so that you can write thing like:
```python
child1.createChild("...")
```

It is also possible to write python expression attached to given component by omitting the "
as in:
```css
Node : {
        name : str(random.random())
}
```

All the python code is executed at load time but stored *un-executed* in the scene graph. Thus it can
be easily modified, re-run and saved. Storing the python fragment in the scene graph also permit to
implement powerful feature as *templates*.


### Templates
In PSL a Template is a component that stores a sub-graph in its textual, or parsed, form. A template
can be instantiated multiple time in the scene graph.
```css
Node : {
    name : "root"
    Template : {
        name : "MyTemplate"
        properties : {
            aName : "undefined"
            numpoints : 3
        }
        Node : {
            name : p"aName"
            MechanicalObject: {
                position : srange(0, numpoints* 3)
            }
            UniformMass : {}
            Node : {
                name : "visual"
                OglModel : { filename : "myOBJ.obj"}
            }
        }
    }

    /// The template can then be instantiated using its name as in:
    MyTemplate : {
        aName : "defined1"
        numpoints : 10
    }

    MyTemplate : {
        aName : "defined2"
        numpoints : 100
    }

    /// Or using Python
    Python : '''
             for i in range(0,10):
                instantiate(root, "MyTemplate", {name:"defined"+str(i), numpoints : i})
             '''
}
```

### Import
To allow template re-usability it is possible to store them in file or directories that can be imported with the Import directive.
In a file named mylibrary.psl" define  a template
```hjson
        Template : { name : "MotorA" ... }
        Template : { name : "MotorB" ... }
        Template : { name : "MotorC" .... }
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

It is also possible to write the template in pure python (TODO- make a link to the right section).

### Aliasing
In Sofa the aliasing system is implicit and the alias are defined in the sofa code source. This is really trouble some as users need to *discover* that in a scene "Mesh" is in fact an alias to a "MeshTopology" object. Without proper tools the solution is often to search in the source code which was an alias.

In PSL we are preserving the use of Alias but we make them explicit. So each scene can defined its own set of alias and anyone reading the scene knows what are the alias and what are the real underlying objects.
```hjson
        Import : mylibrary

        Using : TSPhereModel as CollisionSphere
        Using : mylibrary.MotorA as MotorA
        Using : mylibrary.MotorB as MotorB

        /// Now we can use either
        TSPhereModel : {}

        /// or
        CollisionSphere : {}
```


### Properties
In PSL it is possible to add custom Data field to any sofa object. This is done via the keyword
"properties" that you can use in the following way:

```css
Node : {
    name : "root"

    /// properties are Data field attached to the object at load time.
    properties :
    {
        aIntProperty : 1
        aStringProperty : "Success"
    }
}
```

The type of the poperty is deduced from the type of the data provided among Integer, String and Float
or an array of these.

# Advanced features
## Dynamic templates & GUI interaction
TODO
Template are re-instanciated if their input changed. This imply editting the source code of the
template or editting the values in the GUI.

### Pure Python DSL
To make the writing of python fragment as well as *.pyscn* more elegant we also implemented some
helper function in python.

Example of use in a *.pyscn* file:
```python
    from psl.dsl import *

    createScene(root):
        c = Node(root, name="child1")
        o = MechanicalObject(c, name="mstate)
```

## Writing templates in pure python.
It is also possible write psl template in python file for easier integration with python
oriented workflow. Two possibilities exists:

From a file mytemplate.py:
```python
    PSLExport=["Template1", "Template2"]

    def Template1(name="undefined", numchild=3):
        for i in range(0, numchild):
            c=self.createNode("Child"+str(i))
            c.createObject("MechanicalObject")

    def Template2(name="undefined", numchild=3):
        for i in range(0, numchild):
            c=self.createNode("Child"+str(i))
            c.createObject("MechanicalObject")
```


Alternatively it is possible to use our helper dsl for python to make the writing of
mytemplate.py more elegant:
```python
    @psltemplate
    def Template1(name="undefined", numchild=3):
        for i in range(0, numchild):
            n = Node(self, "Child"+str(i))
            o = MechanicalObjec(n)

    @psltemplate
    def Template2(name="undefined", numchild=3):
        for i in range(0, numchild):
            n = Node(self, "Child"+str(i))
            o = MechanicalObject(n)
```

## Extending-PSL
PSL is a work in progress, if you think about additional great features to add you can submit a
proposal for that. All the proposals we have are stored in the [PEP directory](./pep/).

### Writing PSL scene with XML syntax
PSL is defined in term of an abstract syntax. This syntax can be serialized in multiple concrete syntax.
In this document we have used the HJSON syntax but other alternative exists. One that is interestin is the
XML one (file with a .pslx extensions).
Our initial PSL example now just look like:
```xml
<Node name="myNameisRoot">
        <Node name="child1">
                <MechanicalObject name="mstate"/>
                <OglModel filename="anObj.obj"/>
        </Node>
        <Python>
                Sofa.msg_info(myNameIsRoot, "Hello world")
                for i in range(0,10):
                        myNameIsRoot.addChild("three")
        </Python>
</Node>
```

The interesting aspect of this pslx syntax is that it offer a very good backward compatibility with
existing scene and allow PSL to be integrated into an XML oriented workflow.


