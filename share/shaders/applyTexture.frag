#version 120

varying vec2 TexCoords;

uniform sampler2D Sampler;

void main()
{
    vec4 textureColor = texture2D(Sampler, TexCoords);

    gl_FragColor = vec4(textureColor.rgb, 1.0);
}
