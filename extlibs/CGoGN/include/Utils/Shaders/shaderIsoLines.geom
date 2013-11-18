// ShaderIsoLines::geometryShaderText
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform mat4 ModelViewProjectionMatrix;
uniform vec4 colorMin;
uniform vec4 colorMax;
uniform float vmin;
uniform float vmax;
uniform int vnb;

VARYING_IN float  attribData[3];
VARYING_OUT vec4 ColorFS;


float bary(float x, float xmin, float xmax)
{
	return (x-xmin)/(xmax-xmin);
}


/*
* warning works only with triangles
*/
void isoLine(float x)
{
	ColorFS = mix(colorMin,colorMax,(x-vmin)/(vmax-vmin));

	float b01 = bary(x,attribData[0],attribData[1]);
	float b02 = bary(x,attribData[0],attribData[2]);
	float b12 = bary(x,attribData[1],attribData[2]);
	
	bool in01 = (b01>=0.0) && (b01<=1.0);
	bool in02 = (b02>=0.0) && (b02<=1.0);
	bool in12 = (b12>=0.0) && (b12<=1.0);
	
	if (in01)
	{
		vec4 pos01 = ModelViewProjectionMatrix * mix(POSITION_IN(0),POSITION_IN(1),b01);	
		if (in02)
		{
		// line 01 - 02
			gl_Position = pos01;
			EmitVertex();
			vec4 pos =  mix(POSITION_IN(0),POSITION_IN(2),b02);
			gl_Position = ModelViewProjectionMatrix *  pos;
			EmitVertex();
			EndPrimitive();
		}
		if (in12)
		{
		// line 01 - 12
			gl_Position = pos01;
			EmitVertex();
			vec4 pos =  mix(POSITION_IN(1),POSITION_IN(2),b12);
			gl_Position = ModelViewProjectionMatrix *  pos;
			EmitVertex();
			EndPrimitive();
		}
	}
	if (in02 && in12)
	{
	// line 12 - 02
		vec4 pos =  mix(POSITION_IN(0),POSITION_IN(2),b02);
		gl_Position = ModelViewProjectionMatrix *  pos;
		EmitVertex();
		pos =  mix(POSITION_IN(1),POSITION_IN(2),b12);
		gl_Position = ModelViewProjectionMatrix *  pos;
		EmitVertex();
		EndPrimitive();
	}
}


void main(void)
{
	float inc = (vmax-vmin)/float(vnb);
	for (int i=0; i<vnb; ++i)
	{
		float v = vmin + float(i)*inc;
		isoLine(v);
	}
}
