#version 120
#extension GL_EXT_geometry_shader4: enable

void main() 
{ 
	gl_FrontColor =vec4(1,1,1,1);
    gl_Position = gl_PositionIn[0]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[1]; 
    EmitVertex(); 
    EndPrimitive();
    
    gl_Position = gl_PositionIn[0]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[2]; 
    EmitVertex();
    EndPrimitive();
    
    gl_Position = gl_PositionIn[0]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[3]; 
    EmitVertex();
    EndPrimitive();
    
    gl_Position = gl_PositionIn[1]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[2]; 
    EmitVertex();
    EndPrimitive();
    
    gl_Position = gl_PositionIn[1]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[3]; 
    EmitVertex();
    EndPrimitive();
    
    gl_Position = gl_PositionIn[2]; 
    EmitVertex(); 
    gl_Position = gl_PositionIn[3]; 
    EmitVertex();
    EndPrimitive();

} 