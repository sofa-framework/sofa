#version 120

varying vec2 TexCoords;
varying vec3 LightPosition;
varying vec3 LightHalfVector;
varying vec3 Normal;

void main()
{
    TexCoords = gl_MultiTexCoord0.st;
    Normal = normalize(gl_NormalMatrix * gl_Normal);
    LightPosition = normalize(gl_LightSource[0].position.xyz);
    LightHalfVector = normalize(gl_LightSource[0].halfVector.xyz);

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
