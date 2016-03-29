#version 120

uniform sampler2DRect AccumulationSampler;
uniform sampler2DRect RevealageSampler;

void main()
{
    vec4 Accumulation = texture2DRect(AccumulationSampler, gl_FragCoord.xy);
    float Revealage = texture2DRect(RevealageSampler, gl_FragCoord.xy).r;
    
    vec3 Color = Accumulation.rgb / max(Accumulation.a, 0.00001);

    gl_FragColor = vec4(Color, 1.0) * (1.0 - Revealage);
}
