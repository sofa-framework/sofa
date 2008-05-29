/*****************************************************************************
 * PROJECTED TETRAHEDRA: FRAGMENT SHADER
 * 	Inputs:  the coordinates of the fragment
 *		 dataFragment : interpolation of the scalar and depth
 *	Outputs: color of the fragment in RGBA
 *	Authors:  Sebastien Barbier, Georges-Pierre Bonneau
 *	Version: 1.0
 *	Date:	 March 2008
******************************************************************************/

#version 120

//FROM the GEOMETRY SHADER

varying vec2 dataFragment;

// transfer function: post-integration
uniform sampler1D myLUT;

uniform vec3 fragmentColor;
uniform float fragmentOpacity;

void main(void){

	//vec4 color   = texture1D(myLUT,dataFragment.x);
	vec4 color =  vec4(fragmentColor, fragmentOpacity);
	float alpha  = 1. - exp(-dataFragment.y*color.a);
	gl_FragColor = vec4(color.rgb*alpha, alpha);
	
}
	
