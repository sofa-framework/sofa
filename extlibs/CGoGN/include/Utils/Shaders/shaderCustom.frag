// ShaderCustom::fragmentShaderText
VARYING_FRAG vec4 ColorFS;
VARYING_FRAG vec3 N;
void main()
{
	gl_FragData[0] = ColorFS;
	gl_FragData[1] = vec4( 0.5*normalize(N)+vec3(0.5), 1.0 );
}
