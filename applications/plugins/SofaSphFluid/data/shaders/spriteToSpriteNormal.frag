#version 120

uniform sampler2D u_depthTexture;
uniform float u_width;
uniform float u_height;
uniform mat4 u_InvProjectionMatrix;
uniform float u_zNear;
uniform float u_zFar;

varying vec2 v_texcoord;

vec3 linearDepth(float depth)
{
	float z = depth;      // fetch the z-value from our depth texture
	float n = u_zNear;                                // the near plane
	float f = u_zFar;                               // the far plane
	float c = (2.0 * n) / (f + n - z * (f - n));  // convert to linear values 
	 
	return vec3(c);                      // linear
}

vec3 uvToEye(vec2 texcoord, float depth)
{
	float x = texcoord.x * 2.0 -1.0;
	float y = texcoord.y * 2.0 -1.0;
	vec4 projectedPos = vec4(x,y,depth,1.0);
	vec4 eyePos = u_InvProjectionMatrix * projectedPos;
	return eyePos.xyz/eyePos.w;
}

vec3 getEyePos(sampler2D depthTexture, vec2 texcoord)
{
	return uvToEye(texcoord, texture2D(u_depthTexture, texcoord).x);
}

void main(void)
{
	float maxdepth = 0.99999;
	vec2 texelSize = vec2(1/u_width, 1/u_height);
	//vec2 texcoord = vec2(gl_FragCoord.x/u_width, gl_FragCoord.y/u_height);
	vec2 texcoord = v_texcoord;
	float depth = (texture2D(u_depthTexture, texcoord).x);
	if(depth > maxdepth)
		discard;

	vec3 posEye = uvToEye(texcoord, depth);

	vec3 ddx = getEyePos(u_depthTexture, texcoord + vec2(texelSize.x, 0)) - posEye;
	vec3 ddx2 = posEye - getEyePos(u_depthTexture, texcoord + vec2(-texelSize.x, 0));
	if(abs(ddx.z) > abs(ddx2.z))
		ddx = ddx2;

	vec3 ddy = getEyePos(u_depthTexture, texcoord + vec2(0,texelSize.y)) - posEye;
	vec3 ddy2 = posEye - getEyePos(u_depthTexture, texcoord + vec2(0,-texelSize.y));
	if(abs(ddy.z) > abs(ddy2.z))
		ddy = ddy2;

	vec3 n = cross(ddx, ddy);
	n=normalize(n);

    // vec3 position = uvToEye(texcoord, depth);
    // vec4 n = vec4(normalize(cross(dFdx(position.xyz), dFdy(position.xyz))), 1.0f);

    gl_FragColor = vec4(n.xyz, 1.0);
}