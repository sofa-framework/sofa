#version 120

// uniform float u_radius;

varying vec2 v_mapping;
varying vec4 v_cameraSpherePos;
varying vec3 v_lightDir;
varying float v_radius;

void main()
{
    //const vec3 lightDir = vec3(0,0,1);
    vec3 lightDir = v_lightDir;
    vec4 diffuseColor = gl_FrontMaterial.diffuse;
    vec4 specularColor = gl_FrontMaterial.specular;
    float shininess = gl_FrontMaterial.shininess;

    vec3 cameraPos;
    vec3 cameraNormal;
    
    float lensqr = dot(v_mapping, v_mapping);
    if(lensqr > 1.0)
        discard;
        
    cameraNormal = vec3(v_mapping, sqrt(1.0 - lensqr));
    cameraPos = (cameraNormal * v_radius) + v_cameraSpherePos.xyz;

    //Diffuse
    float diff = max(dot(lightDir, cameraNormal), 0.0);
    vec4 diffuse = diff * diffuseColor;
    //Specular
    vec3 viewDir = normalize(-cameraPos);
    vec3 reflectDir = normalize(reflect(-lightDir, cameraNormal));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);

    vec4 specular = vec4(specularColor.xyz * spec, 1.0) * gl_LightSource[0].specular ;

    gl_FragColor = diffuse + specular;
    //gl_FragColor = vec4(v_mapping,0,1);  
}