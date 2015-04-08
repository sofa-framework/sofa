// ShaderRadiancePerVertex::fragmentShaderInterpText

#extension GL_EXT_gpu_shader4 : enable // need GLSL v1.30 -> if not available, use a uniform for passing textureSize instead of calling textureSize2D

PRECISION;

VARYING_FRAG vec3 vxPos;
VARYING_FRAG vec3 vxNorm;
VARYING_FRAG vec3 barycentricCoord;

flat in ivec2 vx0TexCoord;
flat in ivec2 vx1TexCoord;
flat in ivec2 vx2TexCoord;

uniform vec3 camera;
uniform sampler2D texture;
uniform int resolution;
uniform float K_tab[NB_COEFS];

FRAG_OUT_DEF;

float F_tab[NB_COEFS];

int index (int l, int m)
{
	return l*(l+1)+m;
}

void compute_P_tab (float t)
{
	F_tab[index(0,0)] = 1;
	for (int l=1; l<= resolution; l++)
	{
		F_tab[index(l,l)] = (1-2*l) * sqrt(1-t*t) * F_tab[index(l-1,l-1)];  // first diago
		F_tab[index(l,l-1)] = t * (2*l-1) * F_tab[index(l-1,l-1)];// second diago
		for (int m=0; m<=l-2; m++)
		{// remaining of the line under the 2 diago
			F_tab[index(l,m)] = t * (2*l-1) / float(l-m) * F_tab[index(l-1,m)] - (l+m-1) / float(l-m) * F_tab[index(l-2,m)];
		}
	}
}

void compute_y_tab (float phi)
{
	for (int l=0; l<= resolution; l++)
	{
		F_tab[index(l,0)] *= K_tab[index(l,0)]; // remove for plotting
	}

	for (int m=1; m<=resolution; m++)
	{
		float cos_m_phi = cos ( m * phi );
		float sin_m_phi = sin ( m * phi );

		for (int l=m; l<=resolution; l++)
		{
			F_tab[index(l,m)] *= sqrt(2.0);
			F_tab[index(l,m)] *= K_tab[index(l,m)];
			F_tab[index(l,-m)] = F_tab[index(l,m)] * sin_m_phi ; // store the values for -m<0 in the upper triangle
			F_tab[index(l,m)] *= cos_m_phi;
		}
	}
}

void set_eval_direction (vec3 v)
{
	compute_P_tab(v.z);

	float phi = 0;
	if ((v.x*v.x+v.y*v.y) > 0.0)
		phi = atan(v.y,v.x); // equiv to atan2 in C++

	compute_y_tab(phi);
}

void main (void)
{
	int size = (textureSize2D(texture,0)).x; // supposed square matrix

	vec3 eyeV = normalize(camera - vxPos); // normalized outgoing line-of-sight vector
	eyeV = 2*dot(vxNorm,eyeV)*vxNorm-eyeV ; // symmetrize

	set_eval_direction(eyeV);

	ivec2 param0 = vx0TexCoord;
	ivec2 param1 = vx1TexCoord;
	ivec2 param2 = vx2TexCoord;

	vec3 color = vec3(0.);

	for(int l=0; l<=resolution; l++)
	{
		for (int m = -l; m<=l; m++)
		{
			// compute texture index
			if (param0.y >= size) // if texture newline
			{
				param0.y -= size ;
				param0.x += 1 ;
			}
			if (param1.y >= size) // if texture newline
			{
				param1.y -= size ;
				param1.x += 1 ;
			}
			if (param2.y >= size) // if texture newline
			{
				param2.y -= size ;
				param2.x += 1 ;
			}

			// get corresponding coef
			vec3 coefLM0 = (texelFetch(texture,param0,0)).rgb;
			vec3 coefLM1 = (texelFetch(texture,param1,0)).rgb;
			vec3 coefLM2 = (texelFetch(texture,param2,0)).rgb;

			// multiply by basis function
			float f = F_tab[index(l,m)];
			color += barycentricCoord.x * coefLM0 * f;
			color += barycentricCoord.y * coefLM1 * f;
			color += barycentricCoord.z * coefLM2 * f;

			param0.y ++ ;
			param1.y ++ ;
			param2.y ++ ;
		}
	}

	FRAG_OUT = vec4(color, 1.0) ;
}
