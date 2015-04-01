// ShaderRadiancePerVertex::geometryShaderText
VARYING_IN vec3 ColorAttrib[];
VARYING_OUT vec3 vxColor;
void main()
{
        gl_Position = POSITION_IN(0);
        vxColor = ColorAttrib[0];
        EmitVertex();
        gl_Position = POSITION_IN(1);
        vxColor = ColorAttrib[1];
        EmitVertex();
        gl_Position = POSITION_IN(2);
        vxColor = ColorAttrib[2];
        EmitVertex();
        EndPrimitive();
}
