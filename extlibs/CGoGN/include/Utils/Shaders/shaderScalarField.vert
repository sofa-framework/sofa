// ShaderScalarField::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE float VertexScalar;
uniform mat4 ModelViewProjectionMatrix;
uniform float minValue;
uniform float maxValue;
uniform int colorMap;
uniform int expansion;
VARYING_VERT vec3 color;
VARYING_VERT float scalar;
INVARIANT_POS;

#define M_PI 3.1415926535897932384626433832795

float scale_and_clamp_to_0_1(float x, float min, float max)
{
	float v = (x - min) / (max - min);
	return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

float scale_expand_within_0_1(float x, int n)
{
	for (int i = 1; i <= n; i++)
		x = (1.0 - cos(M_PI * x)) / 2.0;
	for (int i = -1; i >= n; i--)
		x = acos(1.0 - 2.0 * x) / M_PI;
	return x;
}

float scale_expand_towards_1(float x, int n)
{
	for (int i = 1; i <= n; i++)
		x = sin(x * M_PI / 2.0);
	for (int i = -1; i >= n; i--)
		x = asin(x) * 2.0 / M_PI;
	return x;
}

vec3 color_map_blue_white_red(float x)
{
	vec3 c = vec3(0);
	if (x < 0.0)
		c.b = 1.0;
	else if (x < 0.5)
	{
		c.r = 2.0 * x;
		c.g = 2.0 * x;
		c.b = 1.0;
	}
	else if (x < 1.0)
	{
		c.r = 1.0;
		c.g = 2.0 - 2.0 * x;
		c.b = 2.0 - 2.0 * x;
	}
	else
		c.r = 1.0;
	return c;
}

vec3 color_map_cyan_white_red(float x)
{
	if (x < 0.0)
		return vec3(0.0, 0.0, 1.0) ;

	if (x < 0.5)
		return vec3(2.0 * x, 1.0 , 1.0);

	if (x < 1.0)
		return vec3(1.0, 2.0 - 2.0 * x, 2.0 - 2.0 * x);

	return vec3(1.0, 0.0, 0.0) ;
}

vec3 color_map_BCGYR(float x)
{
	if (x < 0.0)
		return vec3(0.0, 0.0, 1.0) ;

	if (x < 0.25)
		return vec3(0.0, 4.0 * x, 1.0);

	if (x < 0.5)
		return vec3(0.0, 1.0 , 2.0 - 4.0 * x);

	if (x < 0.75)
		return vec3(4.0 * x - 2.0, 1.0, 0.0);

	if (x < 1.0)
		return vec3(1.0, 4.0 - 4.0 * x, 0.0);

	return vec3(1.0, 0.0, 0.0) ;
}

vec3 color_map_blue_green_red(float x)
{
	if (x < 0.0)
		return vec3(0.0, 0.0, 1.0) ;

	if (x < 0.5)
		return vec3(0.0, 2.0 * x, 1.0 - 2.0 * x);

	if (x < 1.0)
		return vec3(2.0 * x - 1.0, 2.0 - 2.0 * x, 0.0);

	return vec3(1.0, 0.0, 0.0) ;
}

void main ()
{
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
	float value =
		scale_expand_within_0_1(
			scale_and_clamp_to_0_1(
				VertexScalar,
				minValue,
				maxValue
			),
			expansion
		);

	switch(colorMap)
	{
		case 0 : color = color_map_blue_white_red(value); break;
		case 1 : color = color_map_cyan_white_red(value); break;
		case 2 : color = color_map_BCGYR(value); break;
		case 3 : color = color_map_blue_green_red(value); break;
	}

	scalar = value;
}
