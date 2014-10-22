#version 430 core

// in
attribute vec4 vPosition;

// out
layout(location = 0) out vec2 TexCoords;

void main()
{
    TexCoords = vPosition.xy * 0.5 + 0.5;

    gl_Position = vPosition;
}
