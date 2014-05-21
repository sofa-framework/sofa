#version 430 core

// in
layout(location = 0) in vec2    iTexCoords;

// out
layout(location = 0) out vec4   oColor;

// uniform
uniform float Time;

void main()
{
    lowp float i = 1.0 - (pow(abs(iTexCoords.x), 4.0) + pow(abs(iTexCoords.y), 4.0));
    i = smoothstep(Time - 0.8, Time + 0.8, i);
    i = floor(i * 20.0) / 20.0;

    oColor = vec4(iTexCoords * 0.5 + 0.5, i, i);
    //oColor = vec4(iTexCoords * 0.5 + 0.5, 0, 1.0);
}
