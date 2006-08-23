varying vec3 position;
varying vec4 vertex;
varying vec3 normal;

void main()
{
    position = gl_Vertex.xyz;

    vertex = gl_ModelViewMatrix * gl_Vertex;
    normal = gl_NormalMatrix    * gl_Normal;

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_TexCoord[1] = gl_TextureMatrix[1] * vertex;
   
    gl_Position = ftransform();
}