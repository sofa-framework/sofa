uniform sampler2D       tex;
uniform sampler2DShadow shadowMap;


varying vec3 position;
varying vec4 vertex;
varying vec3 normal;

void main()
{
    // Look up the diffuse color and shadow states for each light source.

    vec4  Kd = texture2D   (tex, gl_TexCoord[0].xy);
    
    float s0 = shadow2DProj(shadowMap, gl_TexCoord[1]).r;
   
       

   if(Kd.x == 0 && Kd.y == 0 && Kd.z == 0)
   Kd = vec4(1, 1, 1, 1);
  
    // Look up the light masks for the spot light sources.

 
    // Compute the lighting vectors.

    vec3 N  = normalize(normal);
    vec3 light_pos = gl_LightSource[0].position.xyz - vertex.xyz;
    vec3 L0 = normalize(light_pos);
 
    // Compute the illumination coefficient for each light source.

    vec3  d0 = vec3(max(dot(N, L0), 0.0)) * s0;
  
    // Compute the scene foreground/background blending coefficient.

    float fade = 1.0;// - smoothstep(48.0, 64.0, length(position));

    // Compute the final pixel color from the diffuse and ambient lighting.

    gl_FragColor = vec4(Kd.rgb * (gl_LightSource[0].diffuse.rgb * d0 +
                                  gl_LightModel.ambient.rgb), Kd.a * fade);
                                  
    // gl_FragColor =    shadow2DProj(shadowMap, gl_TexCoord[1]);
}