//Strings3D::fragmentShaderText1

VARYING_FRAG vec2 tex_coord;
uniform sampler2D FontTexture;
uniform vec4 color;
FRAG_OUT_DEF;
void main (void)
{
	float lum = TEXTURE2D(FontTexture, tex_coord).s;;
// no } because it is added in the shader class code (with other things)
