#version 120

// TO the GEOMETRY SHADER
varying float field;

void main(void){

	field 	       = gl_MultiTexCoord0.r;
	// gl_Position    = ftransform();
	vec4 p = gl_ModelViewProjectionMatrix * gl_Vertex;
	p.xyz *= 1.0/p.w;
	p.w = (gl_ModelViewMatrix * gl_Vertex).z; // keep the original depth in w
	gl_Position    = p;
}
