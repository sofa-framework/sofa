#version 120

varying vec3 LightPosition;
varying vec3 LightHalfVector;
varying vec3 Normal;

void main()
{
    Normal = normalize(gl_NormalMatrix * gl_Normal);
    LightPosition = normalize(gl_LightSource[0].position.xyz);
    LightHalfVector = normalize(gl_LightSource[0].halfVector.xyz);

    gl_FrontColor = gl_Color;
    gl_BackColor = gl_Color;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
