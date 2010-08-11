uniform float shadowBias; 
uniform float zFar;
uniform float zNear;

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
	float depthSqr = dot(lightVec, lightVec) + shadowBias;
	float depth = sqrt(depthSqr);
	//depth = m_depth;
	depth = (depth - zNear)/(zFar - zNear);
	vec4 moments = vec4(0.0,0.0,0.0,0.0);
	moments.xy = ComputeMoments(depth);
	gl_FragColor = moments;
}

