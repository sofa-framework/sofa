#version 120

varying vec2 v_texcoord;

void main(void)
{
	v_texcoord = (gl_Vertex.xy + 1.0) / 2.0;
    gl_Position = ftransform();
}