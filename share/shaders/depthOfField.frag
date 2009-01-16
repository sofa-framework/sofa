#version 120
uniform sampler2D depthTexture, colorTexture;
uniform float blurIntensity;
uniform float focusDistance, focusLength;
varying vec3 vertexPosition;
varying vec2 texCoord;

const float zBase = 0.001;

#define KERNEL_SIZE 12

vec2 offsets[KERNEL_SIZE] = vec2[](
        vec2( -0.326212, -0.405805 ),
        vec2( -0.840144, -0.073580 ),
        vec2( -0.695914,  0.457137 ),
        vec2( -0.203345,  0.620716 ),
        vec2(  0.962340, -0.194983 ),
        vec2(  0.473434, -0.480026 ),
        vec2(  0.519456,  0.767022 ),
        vec2(  0.185461, -0.893124 ),
        vec2(  0.507431,  0.064425 ),
        vec2(  0.896420,  0.412458 ),
        vec2( -0.321940, -0.932615 ),
        vec2( -0.791559, -0.597705 ) );
        
void main()
{
	float zFragment = texture2D(depthTexture , texCoord).x;
	
	vec4 avgColor = texture2D(colorTexture , texCoord);
	int accumulation = 0;
	float coeffDist ;
	if (zFragment < focusDistance)
			coeffDist = abs(zFragment - focusDistance);
	else 
		coeffDist = abs(zFragment - (focusDistance + focusLength));
			
	if (zFragment < focusDistance || zFragment > focusDistance + focusLength)
	{ 
		for (int i = 0 ; i<KERNEL_SIZE; i++)
		{
			vec2 decalage = offsets[i]  * ( coeffDist * blurIntensity);
			
			float zFragmentDecal=texture2D(depthTexture , texCoord + decalage).x;
			
			if (zFragmentDecal < focusDistance || zFragmentDecal > focusDistance + focusLength)
			{
				avgColor += texture2D(colorTexture , texCoord + (decalage));
				accumulation++;
			}
		}
	
		gl_FragColor = avgColor/accumulation;
	}
	else gl_FragColor = avgColor;
	
	//gl_FragColor = vec4(focusDistance,focusDistance,focusDistance,1.0);
	
} 
