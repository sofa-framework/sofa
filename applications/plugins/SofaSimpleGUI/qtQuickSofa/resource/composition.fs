uniform sampler2D   uTexture;

varying highp vec2 coords;

void main()
{
    gl_FragColor = texture(uTexture, coords);
}
