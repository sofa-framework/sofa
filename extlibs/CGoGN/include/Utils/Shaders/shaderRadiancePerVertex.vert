// ShaderRadiancePerVertex::vertexShaderText

#extension GL_EXT_gpu_shader4 : enable // need GLSL v1.30 -> if not available, use a uniform for passing textureSize instead of calling textureSize2D

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexNormal;
ATTRIBUTE ivec2 VertexParam;
uniform mat4 ModelViewProjectionMatrix ;
uniform sampler2D texture;
uniform int resolution;
uniform vec3 camera;
uniform float K_tab[NB_COEFS] ;
VARYING_VERT vec3 ColorAttrib;
INVARIANT_POS;

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

void main ()
{
    int size = (textureSize2D(texture,0)).x; // supposed square matrix

    //init_K_tab();
    vec3 eyeV = normalize(camera - VertexPosition); // normalized outgoing line-of-sight vector
	eyeV = 2*dot(VertexNormal,eyeV)*VertexNormal-eyeV ; // symmetrize
    set_eval_direction(eyeV);

    ColorAttrib = vec3(0.,0.,0.) ;

    // evaluate function
    ivec2 param = VertexParam ;
    for(int l=0; l<=resolution; l++)
    {
        for (int m = -l; m<=l; m++)
        {
            // compute texture index
            if (param.y >= size) // if texture newline
            {
                param.y -= size ;
                param.x += 1 ;
            }

            // get corresponding coef
            vec3 coefLM = (texelFetch(texture,param,0)).rgb;
            // multiply by basis function
            ColorAttrib += coefLM * F_tab[index(l,m)] ;

            param.y ++ ;
        }
    }

    // ColorAttrib = eyeV ; // Debug camera position
    // ColorAttrib = VertexNormal ; // Debug normals
    // ColorAttrib = (VertexPosition+vec3(1.,1.,1.))/2.0 ; // Debug positions
    gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
