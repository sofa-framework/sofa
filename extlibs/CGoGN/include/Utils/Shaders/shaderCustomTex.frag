// ShaderCustomTex::fragmentShaderText

PRECISON;
uniform sampler2D textureUnit;
in vec3 N;
in vec2 fragTexCoord;
uniform vec4 ambient;
FRAG_OUT_DEF;
void main()
{
	gl_FragData[0] = ambient*texture2D(textureUnit,fragTexCoord);
	gl_FragData[1] = vec4( 0.5*normalize(N)+vec3(0.5), 1.0 );
}