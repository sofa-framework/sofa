#version 130
#extension GL_EXT_geometry_shader4: enable

//GLSL >= 130
uniform samplerBuffer u_barycenter_tex;
//uniform sampler2D u_barycenter_tex;

uniform float volumeScale;

//GLSL >= 130
in vec4 volumeColor[4]; 
in mat4 matproj[4]; 
out vec3 triangleNormal;

//varying in vec4 volumeColor[4]; 
//varying in mat4 matproj[4]; 
//varying out vec3 triangleNormal;

void main() 
{ 
    //vec4 barycenter = (gl_PositionIn[0]+gl_PositionIn[1]+gl_PositionIn[2]+gl_PositionIn[3])*0.25;

    //GLSL >= 130
    vec3 texBarycenter = texelFetch(u_barycenter_tex, gl_PrimitiveIDIn).xyz;
    //vec3 texBarycenter = texture2D(u_barycenter_tex, vec2(gl_PrimitiveIDIn,0)).xyz;

    vec4 barycenter = matproj[0] * vec4(texBarycenter, 1.0);

    float currentVolumeScale = 1.0;
    if(volumeScale > 0.0 || volumeScale < 1.0)
        currentVolumeScale = volumeScale;

    vec4 newPositionIn[4];
    newPositionIn[0] = (gl_PositionIn[0] - barycenter)*currentVolumeScale + barycenter; 
    newPositionIn[1] = (gl_PositionIn[1] - barycenter)*currentVolumeScale + barycenter; 
    newPositionIn[2] = (gl_PositionIn[2] - barycenter)*currentVolumeScale + barycenter; 
    newPositionIn[3] = (gl_PositionIn[3] - barycenter)*currentVolumeScale + barycenter; 

    ////Generate triangles
    //Triangle 0
    vec4 d1 = gl_PositionIn[3] - gl_PositionIn[0];
    vec4 d2 = gl_PositionIn[2] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor = volumeColor[0];
    gl_ClipVertex = gl_ClipVertexIn[0];
    gl_Position = newPositionIn[0];
    EmitVertex(); 
    gl_FrontColor = volumeColor[2];
    gl_ClipVertex = gl_ClipVertexIn[2];
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    gl_FrontColor = volumeColor[3];
    gl_ClipVertex = gl_ClipVertexIn[3];
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 1
    d1 = gl_PositionIn[2] - gl_PositionIn[0];
    d2 = gl_PositionIn[1] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor =volumeColor[0];
    gl_ClipVertex = gl_ClipVertexIn[0];
    gl_Position = newPositionIn[0]; 
    EmitVertex(); 
    gl_FrontColor = volumeColor[1];
    gl_ClipVertex = gl_ClipVertexIn[1];
    gl_Position = newPositionIn[1];
    EmitVertex(); 
    gl_FrontColor = volumeColor[2];
    gl_ClipVertex = gl_ClipVertexIn[2];
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 2
    d1 = gl_PositionIn[1] - gl_PositionIn[0];
    d2 = gl_PositionIn[3] - gl_PositionIn[0];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor =volumeColor[0];
    gl_ClipVertex = gl_ClipVertexIn[0];
    gl_Position = newPositionIn[0]; 
    EmitVertex(); 
    gl_FrontColor = volumeColor[1];
    gl_ClipVertex = gl_ClipVertexIn[1];
    gl_Position = newPositionIn[1];
    EmitVertex(); 
    gl_FrontColor = volumeColor[3];
    gl_ClipVertex = gl_ClipVertexIn[3];
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();
    
    //Triangle 3
    d1 = gl_PositionIn[2] - gl_PositionIn[1];
    d2 = gl_PositionIn[3] - gl_PositionIn[1];
    triangleNormal = cross(d1.xyz,d2.xyz);
    gl_FrontColor = volumeColor[1];
    gl_ClipVertex = gl_ClipVertexIn[1];
    gl_Position = newPositionIn[1]; 
    EmitVertex(); 
    gl_FrontColor = volumeColor[2];
    gl_ClipVertex = gl_ClipVertexIn[2];
    gl_Position = newPositionIn[2];
    EmitVertex(); 
    gl_FrontColor = volumeColor[3];
    gl_ClipVertex = gl_ClipVertexIn[3];
    gl_Position = newPositionIn[3];
    EmitVertex(); 
    EndPrimitive();

} 