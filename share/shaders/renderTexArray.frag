#version 120
#extension GL_EXT_gpu_shader4 : enable

//3D Texture Sampler
uniform sampler2DArray samp;
uniform int	layerDepth;

void main(void)
{
	gl_FragColor = texture2DArray( samp, vec3(gl_TexCoord[0].xy, layerDepth));
}