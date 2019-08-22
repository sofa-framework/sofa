#version 120
uniform sampler2D depthTexture, colorTexture;
uniform vec2 pixelSize;

uniform float blurIntensity;
uniform float focusDistance, focusLength;
uniform int showDepthMap;


varying vec3 vertexPosition;
varying vec2 texCoord;

#define RHO 2.0
#define NB_PIXEL_STEP 2
        
float getGaussianDistribution( float x, float y, float rho )
{
   float g = 1.0f / sqrt( 2.0f * 3.141592654f * rho * rho );
   return g * exp( -(x * x + y * y) / (2 * rho * rho) );
}     
        
void main()
{
	float zFragment = texture2D(depthTexture , texCoord).x;
	
	vec4 originalColor = texture2D(colorTexture , texCoord);
	vec4 avgColor = originalColor;
	
	if (pixelSize.x == 0.0 || pixelSize.y == 0.0)
	{
		gl_FragColor = originalColor;
		return;
	} 
	
	vec4 finalColor;
	float accumulation = 0;
	int test=0;
	float zDistance ;
	if (zFragment < focusDistance)
			zDistance = focusDistance - zFragment;
	else if (zFragment > focusDistance + focusLength)
		zDistance = zFragment - (focusDistance + focusLength);
	else zDistance = 0.0;
	
	float r =  zDistance * blurIntensity;

	//if ( zDistance > 0.0)
	{ 
		avgColor=vec4(0.0,0.0,0.0,0.0);
		float i=texCoord.x - r;
		while( i<( texCoord.x + r) )
		{
			float j=texCoord.y - r;
			while( j< ( texCoord.y + r ) )
			{

					avgColor += texture2D(colorTexture , vec2(i,j)) * getGaussianDistribution( i - texCoord.x,j - texCoord.y ,RHO);
					accumulation += getGaussianDistribution( i - texCoord.x,j - texCoord.y ,RHO);
				
				
				j += (pixelSize.y*NB_PIXEL_STEP);
			}
			i += (pixelSize.x*NB_PIXEL_STEP);
		}
	}
	
	if (accumulation > 0.0)
		finalColor = avgColor*(1/accumulation);
	else finalColor = originalColor;
	
	if (showDepthMap > 0.0)
		finalColor = vec4(zFragment, zFragment, zFragment, 1.0);
	
	gl_FragColor = finalColor;;
	
	//if (test > 1)
	//	gl_FragColor = vec4(accumulation,accumulation,accumulation,1.0);
	
} 
