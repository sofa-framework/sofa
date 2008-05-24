/*****************************************************************************
 * PROJECTED TETRAHEDRA: VERTEX SHADER
 * 	Inputs:  the coordinates of the vertices 
 *		 the scalar field into the red channel of the texture
 *	Outputs: gl_Position, the position of the vertices in image space
 *		 varying field, the scalar field
 *	Authors: Sebastien Barbier, Georges-Pierre Bonneau
 *	Version: 1.0
 *	Date:	 March 2008
******************************************************************************/

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
