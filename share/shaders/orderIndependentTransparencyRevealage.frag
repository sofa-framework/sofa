#version 120

varying vec3 LightPosition;
varying vec3 LightHalfVector;
varying vec3 Normal;

void main()
{
    float finalAlpha = gl_FrontMaterial.diffuse.a;
    //finalAlpha = gl_Color.a; // TODO: remove this

    gl_FragData[0] = vec4(finalAlpha);
}
