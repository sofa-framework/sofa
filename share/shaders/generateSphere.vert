#version 120

attribute float a_radius;

// uniform float u_radius;

varying vec2 v_mapping;
varying vec4 v_cameraSpherePos;
varying vec3 v_lightDir;
varying float v_radius;

void main(void)
{
	vec2 offset;

	v_cameraSpherePos = (gl_ModelViewMatrix * gl_Vertex);

	gl_TexCoord[0] = gl_MultiTexCoord0;

	v_mapping = vec2(gl_TexCoord[0].x , gl_TexCoord[0].y);
	offset = v_mapping * a_radius;
    
    vec4 cameraCornerPos = v_cameraSpherePos;
    cameraCornerPos.xy += offset;

    vec3 vertexToLightSource = gl_LightSource[0].position.xyz - cameraCornerPos.xyz;
    v_lightDir = normalize(vertexToLightSource);  
    v_radius = a_radius; 	
    
    gl_Position = gl_ProjectionMatrix * cameraCornerPos;
    // gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

}
