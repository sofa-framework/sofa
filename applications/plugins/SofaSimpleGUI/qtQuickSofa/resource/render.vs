attribute highp vec4 vertices;

varying highp vec2 coords;

void main()
{
    gl_Position = vertices;
    coords = vertices.xy;
}
