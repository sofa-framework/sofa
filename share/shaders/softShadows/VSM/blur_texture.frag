uniform sampler2D colorTexture;
uniform float mapDimX;
uniform int orientation; // 0 -> Horizontal, 1 -> Vertical

const float M_PI = 3.14159265358979323846;
const float M_E = 2.71828182845904523536;

void gaussian_filter_kernel(out float gauss[16], int size, float sigma)
{
    if (sigma == 0.0)
    {
        gauss[0] = 1.0;
        for (int i = 1; i < 16; i++)
            gauss[i] = 0.0;
    }
    else
    {
        int i;

        float len = 0.0;
        for (i = 0; i < size; i++)
        {
            float a = 1.0 / (2.0*M_PI*sigma*sigma) * pow(M_E, float((-i*i))/(2.0*sigma*sigma));
            gauss[i] = a;
            len += (i != 0)? a : 2.0*a;
        }

        len = 1.0/len;
        for (i = 0; i < size; i++)
            gauss[i] *= len;

        for (i = size; i < 16; i++)
            gauss[i] = 0.0;
    }
}

vec4 tex2Dgauss_partial(sampler2D tex, vec2 texcoord, vec2 dim, int size, float gauss[16])
{
    vec4 result = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < size*2-1; i++)
    {
	int index = int(abs(float(i-size+1))); //abs() wants float ...
        result += texture2D(tex, texcoord + vec2(i-size+1, i-size+1) * dim) * gauss[index];
    }
    return result;
}

vec4 tex2DgaussN(sampler2D tex, vec2 texcoord, vec2 dim, int size, float sigma)
{
    float gauss[16];
    gaussian_filter_kernel(gauss, size, sigma);
    return tex2Dgauss_partial(tex, texcoord, dim, size, gauss);
}

vec4 tex2Dgauss3(sampler2D tex, vec2 texcoord, vec2 dim)  { return tex2DgaussN(tex, texcoord, dim, 2 , 0.7); }
vec4 tex2Dgauss5(sampler2D tex, vec2 texcoord, vec2 dim)  { return tex2DgaussN(tex, texcoord, dim, 3 , 0.9); }
vec4 tex2Dgauss7(sampler2D tex, vec2 texcoord, vec2 dim)  { return tex2DgaussN(tex, texcoord, dim, 4 , 1.1); }
vec4 tex2Dgauss9(sampler2D tex, vec2 texcoord, vec2 dim)  { return tex2DgaussN(tex, texcoord, dim, 5 , 1.4); }
vec4 tex2Dgauss11(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 6 , 1.8); }
vec4 tex2Dgauss13(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 7 , 2.2); }
vec4 tex2Dgauss15(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 8 , 2.6); }
vec4 tex2Dgauss19(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 10, 3.5); }
vec4 tex2Dgauss23(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 12, 4.5); }
vec4 tex2Dgauss27(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 14, 5.5); }
vec4 tex2Dgauss31(sampler2D tex, vec2 texcoord, vec2 dim) { return tex2DgaussN(tex, texcoord, dim, 16, 6.6); }

void main()
{
	vec2 texcoord = gl_TexCoord[0].xy;
	vec4 color = texture2D(colorTexture, texcoord);
	vec2 dim;
	if(orientation == 0)
		dim = vec2(0, 1.0/mapDimX);
	else 
		dim = vec2(1.0/mapDimX, 0);
	color = tex2Dgauss7(colorTexture, texcoord, dim);
	gl_FragColor = color;
}

