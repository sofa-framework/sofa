#version 420 compatibility

struct V2T
{
    vec3 position;
    //vec3 normal;
};

struct TC2E
{
    vec3 position;
    //vec3 normal;
};

struct T2G
{
    vec3 position;
    //vec3 normal;
    vec3 patchDistance;
};

struct G2F
{
    vec3 position;
    vec3 normal;
    vec3 patchDistance;
    vec3 triDistance;
};

#ifdef VertexShader //--------------------------------------
out V2T v;

void main()
{
    v.position = gl_Vertex.xyz;
}

#endif
#ifdef TessellationEvaluationShader //-----------------------
layout(triangles, equal_spacing, cw) in;
in V2T v[];
out vec3 tePosition;
out vec3 tePatchDistance;

void main()
{
    tePatchDistance = gl_TessCoord;
    tePosition = normalize(gl_TessCoord.x * v[0].position + gl_TessCoord.y * v[1].position + gl_TessCoord.z * v[2].position);
    gl_Position = gl_ModelViewProjectionMatrix * vec4(tePosition, 1);
}

#endif
#ifdef GeometryShader //------------------------------------

uniform mat4 Modelview;
uniform mat3 NormalMatrix;
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec3 tePosition[3];
in vec3 tePatchDistance[3];
out vec3 gPosition;
out vec3 gNormal;
out vec3 gPatchDistance;
out vec3 gTriDistance;

void main()
{
/*
    vec3 A = tePosition[2] - tePosition[0];
    vec3 B = tePosition[1] - tePosition[0];
    gNormal = normalize(cross(A, B));
*/
    gPosition = tePosition[0];
    gNormal = gPosition;
    gPatchDistance = tePatchDistance[0];
    gTriDistance = vec3(1, 0, 0);
    gl_Position = gl_in[0].gl_Position; EmitVertex();

    gPosition = tePosition[1];
    gNormal = gPosition;
    gPatchDistance = tePatchDistance[1];
    gTriDistance = vec3(0, 1, 0);
    gl_Position = gl_in[1].gl_Position; EmitVertex();

    gPosition = tePosition[2];
    gNormal = gPosition;
    gPatchDistance = tePatchDistance[2];
    gTriDistance = vec3(0, 0, 1);
    gl_Position = gl_in[2].gl_Position; EmitVertex();

    EndPrimitive();
}
#endif
#ifdef FragmentShader //------------------------------------
//out vec4 FragColor;
in vec3 gPosition;
in vec3 gNormal;
in vec3 gTriDistance;
in vec3 gPatchDistance;
//uniform vec3 LightPosition;
//uniform vec3 DiffuseMaterial;
//uniform vec3 AmbientMaterial;

float amplify(float d, float scale, float offset)
{
    d = scale * d + offset;
    d = clamp(d, 0, 1);
    d = 1 - exp2(-2*d*d);
    return d;
}

void main()
{
    vec3 N = normalize(gNormal);
    vec3 L = normalize(vec3(100,0,0) - gPosition);
    float df = max(0,dot(N, L));
    vec3 color = gl_FrontLightProduct[0].diffuse.rgb * (0.2+df);

    float d1 = min(min(gTriDistance.x, gTriDistance.y), gTriDistance.z);
    float d2 = min(min(gPatchDistance.x, gPatchDistance.y), gPatchDistance.z);
    color = amplify(d1, 40, -0.5) * amplify(d2, 60, -0.5) * color;

    gl_FragColor = vec4(color, 1.0);
}
#endif
