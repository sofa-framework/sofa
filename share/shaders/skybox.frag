#version 120

in vec3 TexCoords; //expected to be between -1 and 1

uniform sampler2D skybox_front; // positive X
uniform sampler2D skybox_back; // negative X

uniform sampler2D skybox_right; // positive Y
uniform sampler2D skybox_left; // negative Y

uniform sampler2D skybox_top; // positive Z
uniform sampler2D skybox_bottom; // negative Z

void main()
{
    vec4 color;

    if (abs(TexCoords.x - 1) < 1e-3)
    {
        color = vec4(texture2D(skybox_front, TexCoords.yz / 2 + 0.5));
    }
    else if (abs(TexCoords.x + 1) < 1e-3)
    {
        vec2 uv = TexCoords.yz / 2 + 0.5;
        uv.x *= -1;
        uv.x += 1;
        color = vec4(texture2D(skybox_back, uv));
    }
    else if (abs(TexCoords.y - 1) < 1e-3)
    {
        vec2 uv = TexCoords.xz / 2 + 0.5;
        uv.x *= -1;
        uv.x += 1;
        color = vec4(texture2D(skybox_right, uv));
    }
    else if (abs(TexCoords.y + 1) < 1e-3)
    {
        color = vec4(texture2D(skybox_left, TexCoords.xz / 2 + 0.5));
    }
    else if (abs(TexCoords.z - 1) < 1e-3)
    {
        vec2 uv = TexCoords.yx / 2 + 0.5;
        uv.y *= -1;
        uv.y += 1;
        color = vec4(texture2D(skybox_top, uv));
    }
    else
    {
        color = vec4(texture2D(skybox_bottom, TexCoords.yx / 2 + 0.5));
    }
    gl_FragColor = clamp(color, 0.0, 1.0);
}