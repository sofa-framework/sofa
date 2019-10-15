#version 120

uniform sampler2D u_normalTexture;
uniform sampler2D u_depthTexture;
uniform sampler2D u_thicknessTexture;

uniform vec4 u_diffuseColor;
uniform mat4 u_InvProjectionMatrix;
uniform float u_width;
uniform float u_height;

varying vec2 v_texcoord;

vec3 uvToEye(vec2 texcoord, float depth)
{
	float x = texcoord.x * 2.0 -1.0;
	float y = texcoord.y * 2.0 -1.0;
	vec4 projectedPos = vec4(x,y,depth,1.0);
	vec4 eyePos = u_InvProjectionMatrix * projectedPos;
	return eyePos.xyz/eyePos.w;
}

void main(void)
{
	const vec3 lightDir = vec3(0,0,1);
	vec4 diffuseColor = u_diffuseColor;

	float depth = texture2D(u_depthTexture, v_texcoord).x;
	if(depth > 0.999f)
		discard;

	vec3 normal = texture2D(u_normalTexture, v_texcoord).xyz;
	normal = normalize(normal);
	vec3 eyePos = uvToEye(v_texcoord, depth);

	//Diffuse
	float diff = max(dot(lightDir, normal), 0.0) *0.5+0.5;
	vec4 diffuse = diff * diffuseColor;
	//Specular
    vec3 viewDir = -eyePos;
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    bool blinn = true;
    if(blinn)
    {
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        spec = pow(max(dot(normal, halfwayDir), 0.0), 1280.0);
    }
    else
    {
        vec3 reflectDir = reflect(-lightDir, normal);
        spec = pow(max(dot(viewDir, reflectDir), 0.0), 8.0);
    }
    vec4 specular = vec4(vec3(1.0,1.0,1.0) * spec, 1.0); // assuming bright white light color

	float thickness = texture2D(u_thicknessTexture, v_texcoord).x;
	float distPwr = 1-thickness;
	vec4 backgroundDistorsion = vec4(1.0,1.0,1.0,1.0);//texture2D(backgroundTexture, v_texcoord + normal.xy * .025 * thickness).xyz;
	diffuse = (1 - distPwr) * diffuse + distPwr * backgroundDistorsion;

	vec4 finalColor = diffuse + specular;

	gl_FragColor = finalColor;

}