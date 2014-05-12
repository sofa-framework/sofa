attribute highp vec4 vertices;

varying highp vec2 coords;

void main()
{
    gl_Position = vertices;
    coords = vertices.xy * 0.5 + 0.5;
}
