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
	
