#version 120

varying vec2 TexCoords;

void main()
{
    TexCoords = gl_MultiTexCoord0.st;
    
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
