#version 430 core

// in
attribute vec4                  vPosition;

// out
layout(location = 0) out vec2   TexCoords;

void main()
{
    TexCoords = vPosition.xy;

    gl_Position = vPosition;
}
