<?xml version="1.0"?>
<!DOCTYPE sml SYSTEM "sml.dtd">

<sml name="bielle_manivelle">
    <units mass="kg" length="dm" time="s" />
    
    <mesh id="2.obj">
        <source format="obj">2.obj</source>
    </mesh>
    <mesh id="3.obj">
        <source format="obj">3.obj</source>
    </mesh>
    
    <solid id="1">
        <name>Corps</name>
        <tag>rigidScale</tag>
        <position>0 0 0 0 0 0 1</position>
        <mesh collision="False" id="0.obj">
            <source format="obj">0.obj</source>
        </mesh>
    </solid>
    
    <solid id="2">
        <name>Roue</name>
        <tag>rigidScale</tag>
        <tag>bigDeformation</tag>
        <position>0 0 -0.148 0 0 0 1</position>
        <mesh id="3.obj" />
    </solid>
    
    <solid id="3">
        <name>Came</name>
        <tag>rigidScale</tag>
        <position>1.085 -0.072 0.33 0 0 0 1</position>
        <mesh id="2.obj" />
    </solid>
    
    <solid id="4">
        <name>Piston</name>
        <tag>rigidScale</tag>
        <position>2.05 0 0.33 0 0 0 1</position>
        <mesh id="1.obj">
            <source format="obj">1.obj</source>
        </mesh>
    </solid>
        
    <jointGeneric id="5">
        <name>hinge_corps-roue</name>
        <jointSolidRef id="1">
            <offset type="relative">0 0 0 0 0 0 1</offset>
        </jointSolidRef>
        <jointSolidRef id="2">
            <offset type="relative">0 0 0.148 0 0 0 1</offset>
        </jointSolidRef>
        <dof index="rz"/>
    </jointGeneric>
    
    <jointGeneric id="6">
        <name>hinge_roue-came</name>
        <jointSolidRef id="2">
            <offset type="relative">0.24 -0.145 0.478 0 0 0 1</offset>
        </jointSolidRef>
        <jointSolidRef id="3">
            <offset type="relative">-0.845 -0.073 0 0 0 0 1</offset>
        </jointSolidRef>
        <dof index="rz"/>
    </jointGeneric>
    
    <jointGeneric id="7">
        <name>hinge_came-piston</name>
        <jointSolidRef id="3">
            <offset type="relative">0.852 0.072 0 0 0 0 1</offset>
        </jointSolidRef>
        <jointSolidRef id="4">
            <offset type="relative">-0.113 0 0 0 0 0 1</offset>
        </jointSolidRef>
        <dof index="rz"/>
    </jointGeneric>
    
    <jointGeneric id="8">
        <name>slider_corps-piston</name>
        <jointSolidRef id="4">
            <offset type="relative">0.15 0 0 0 0 0 1</offset>
        </jointSolidRef>
        <jointSolidRef id="1">
            <offset type="relative">2.2 0 0.33 0 0 0 1</offset>
        </jointSolidRef>
        <dof index="x" min="-0.05" max="0.6"/>
    </jointGeneric>

</sml>