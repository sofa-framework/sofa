#version 120

out vec3 TexCoords;

void main()
{
    TexCoords = vec3(gl_Vertex);
    gl_Position = gl_ModelViewProjectionMatrix * vec4(100 * gl_Vertex.xyz, 1);
}