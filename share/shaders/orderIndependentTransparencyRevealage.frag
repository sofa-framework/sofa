#version 120

varying vec2 TexCoords;

uniform bool HasTexture;
uniform sampler2D ColorSampler;

float ComputeAlpha()
{
    float alpha = gl_FrontMaterial.diffuse.a;
    
    if(HasTexture)
    {
        vec4 colorTexture = texture2D(ColorSampler, TexCoords);
        
        alpha *= colorTexture.a;
    }
    
    return alpha;
}

void main()
{
    float finalAlpha = ComputeAlpha();

    gl_FragData[0] = vec4(finalAlpha);
}
