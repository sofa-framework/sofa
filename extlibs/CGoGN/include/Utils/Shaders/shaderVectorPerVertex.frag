// ShaderVectorPerVertex::fragmentShaderText

PRECISON;
uniform vec4 vectorColor;
FRAG_OUT_DEF;
void main()
{
	gl_FragColor = vectorColor;
}
