#version 120

uniform float u_shadowBias; 
uniform float u_zFar;
uniform float u_zNear;

varying vec4 lightVec;

vec2 ComputeMoments(float depth)
{
    vec2 moments;

    // First moment is the depth itself.
    moments.x = depth*1.02;

    // Compute partial derivatives of depth.
    float dx = dFdx(depth);
    float dy = dFdy(depth);

    // Compute second moment over the pixel extents.
    moments.y = depth*depth + 0.25*(dx*dx + dy*dy);

    return moments;
}

void main()
{
	float depthSqr = dot(lightVec, lightVec) + u_shadowBias;
	float depth = sqrt(depthSqr);
	//depth = m_depth;
	depth = (depth - u_zNear)/(u_zFar - u_zNear); // make it between znear and zfar and linear
    
	vec4 moments = vec4(0.0,0.0,0.0,0.0);
	moments.xy = ComputeMoments(depth);
	gl_FragColor = moments;
}

