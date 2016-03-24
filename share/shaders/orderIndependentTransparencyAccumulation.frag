#version 120

varying vec3 LightPosition;
varying vec3 LightHalfVector;
varying vec3 Normal;

uniform float DepthScale;

void main()
{
    float diffuseIntensity = max(abs(dot(Normal, LightPosition)), 0.0);

    vec3 ambientColor = vec3(gl_FrontMaterial.ambient * gl_LightSource[0].ambient + gl_LightModel.ambient * gl_FrontMaterial.ambient);
    vec3 diffuseColor = vec3(gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse);
    vec3 specularColor = vec3(gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(max(abs(dot(Normal, LightHalfVector)), 0.0), gl_FrontMaterial.shininess));

    vec4 finalColor = vec4(ambientColor + diffuseColor * diffuseIntensity + specularColor, gl_FrontMaterial.diffuse.a);
    //finalColor = gl_Color; // TODO: remove this

    float viewDepth = abs(1.0 / gl_FragCoord.w);

    // Tuned to work well with FP16 accumulation buffers and 0.001 < linearDepth < 2.5
    float linearDepth = viewDepth * DepthScale;
    float weight = clamp(0.03 / (1e-5 + pow(linearDepth, 4.0)), 1e-2, 3e3);

    gl_FragData[0] = vec4(finalColor.rgb * finalColor.a, finalColor.a) * weight;
}
