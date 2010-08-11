varying vec4 lightVec;
varying float m_depth;

uniform vec4 lightPosition;

void main()
{	
	vec4 lp;
	lp = lightPosition; 
	lp = vec4(0.0,0.0,0.0,1.0); 
	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	lightVec = ecPos-lp;

	m_depth = ecPos.z;

	gl_Position = ftransform();
}

