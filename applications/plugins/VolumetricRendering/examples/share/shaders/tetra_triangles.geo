#version 120
#extension GL_EXT_geometry_shader4: enable

uniform float tetraScale;
varying in vec4 tetraColor[4]; //
varying out vec3 triangleNormal;

void main() 
{ 
    vec4 barycenter = (gl_PositionIn[0]+gl_PositionIn[1]+gl_PositionIn[2]+gl_PositionIn[3])*0.25;

    float currentTetraScale = 1.0;
    if(tetraScale > 0.0 || tetraScale < 1.0)
        currentTetraScale = tetraScale;

    vec4 newPositionIn[4];
    newPositionIn[0] = (gl_PositionIn[0] - barycenter)*currentTetraScale + barycenter; 
    newPositionIn[1] = (gl_PositionIn[1] - barycenter)*currentTetraScale + barycenter; 
    newPositionIn[2] = (gl_PositionIn[2] - barycenter)*currentTetraScale + barycenter; 
    newPositionIn[3] = (gl_PositionIn[3] - barycenter)*currentTetraScale + barycenter; 

    ////Generate triangles
    //Triangle 0
    vec4 d1 = gl_PositionIn[3] - gl_PositionIn[0];
    vec4 d2 = gl_PositionIn[2] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
	gl_FrontColor = tetraColor[0];
    gl_Position = newPositionIn[0];
    EmitVertex(); 
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 1
    d1 = gl_PositionIn[2] - gl_PositionIn[0];
    d2 = gl_PositionIn[1] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor =tetraColor[1];
    gl_Position = newPositionIn[0]; 
    EmitVertex(); 
    gl_Position = newPositionIn[1];
    EmitVertex(); 
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 2
    d1 = gl_PositionIn[1] - gl_PositionIn[0];
    d2 = gl_PositionIn[3] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor =tetraColor[2];
    gl_Position = newPositionIn[0]; 
    EmitVertex(); 
    gl_Position = newPositionIn[1];
    EmitVertex(); 
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 3
    d1 = gl_PositionIn[2] - gl_PositionIn[1];
    d2 = gl_PositionIn[3] - gl_PositionIn[1];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor =tetraColor[3];;
	gl_Position = newPositionIn[1]; 
    EmitVertex(); 
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();

} 