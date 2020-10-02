<?xml version="1.0"?>
<!DOCTYPE sml SYSTEM "sml.dtd"> 

<sml name="sml_display">

    <units length="dm" mass="kg" time="s"/>

    <mesh id="ball">
        <source format="obj">mesh/ball.obj</source>
        <group id="group01">
            <index>1 2 3 4</index>
            <tag>test</tag>
        </group>
    </mesh>
    <mesh id="armadillo">
        <source format="obj">mesh/Armadillo_verysimplified.obj</source>
    </mesh>
    
    <solid id="armadillo">
        <tag>red</tag>
        <position>0 0 0 0 0 0 1</position>
        <mesh id="armadillo" />
        
        <offset type="absolute" >0.190269 14.636515 1.719540 0.707107 0.000000 -0.000000 0.707107</offset>
        <offset type="absolute" name="myoffset">1 1 1   0 0 0 1</offset>
        
    </solid>
    
    <solid id="ball01">
        <tag>green</tag>
        <position>0 8 -5 0 0 0 1</position>
        <mesh id="ball" />
    </solid>
    
    <solid id="ball02">
        <position>0 8 -7 0 0 0 1</position>
        <mesh id="ball" />
    </solid>
    
</sml>