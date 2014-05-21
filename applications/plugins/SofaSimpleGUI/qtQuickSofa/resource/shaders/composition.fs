#version 430 core

// in
layout(location = 0) in vec2 TexCoords;

// out
layout(location = 0) out vec4 Color;

// uniform
uniform sampler2D   uTexture;

void main()
{
    Color = texture(uTexture, TexCoords);
    // Color = vec4(1.0, 0.0, 0.0, 1.0);
}
