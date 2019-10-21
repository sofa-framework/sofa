#version 120

uniform mat4 u_projectionMatrix;
uniform float u_zNear;
uniform float u_zFar;
uniform float u_spriteRadius;
uniform float u_spriteThickness;

varying vec4 eyeSpacePos;

vec3 linearDepth(float depth)
{
	float z = depth;      // fetch the z-value from our depth texture
	float n = u_zNear;                                // the near plane
	float f = u_zFar;                               // the far plane
	float c = (2.0 * n) / (f + n - z * (f - n));  // convert to linear values 
	 
	return vec3(c);                      // linear
}

// result suitable for assigning to gl_FragDepth
float depthSample(float linearDepth)
{
    float nonLinearDepth = (u_zFar + u_zNear - 2.0 * u_zNear * u_zFar / linearDepth) / (u_zFar - u_zNear);
    nonLinearDepth = (nonLinearDepth + 1.0) / 2.0;
    return nonLinearDepth;
}

float projectZ(float z)
{
	float near = u_zNear;                                // the near plane
	float far = u_zFar;                               // the far plane
	return far*(z+near)/(z*(far-near));
}

void main(void)
{
    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);  
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

	// calculate depth
	vec4 spherePosEye = vec4(eyeSpacePos.xyz + N*u_spriteRadius, 1.0);
	vec4 clipSpacePos = u_projectionMatrix * spherePosEye;
	float normDepth = clipSpacePos.z / clipSpacePos.w;

	//Transform into window coordinates
	gl_FragDepth = ((gl_DepthRange.diff * normDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
    // Thickness
    float thickCoeff= u_spriteThickness;
	vec4 thickness = vec4(thickCoeff,thickCoeff,thickCoeff, 1);

    //float diffuse = max(0.0, dot(lightDir, N));
    //gl_FragColor = vec4(Color,1) * diffuse;
    //gl_FragColor = vec4(linearDepth(normDepth), 1.0);
    //Better to put in another color_attachmentX = gl_FragData[X]
	gl_FragColor = thickness;
}