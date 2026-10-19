// Verbatim copy of Example49/shader/background.frag.glsl (McNopper/OpenGL v3.2.7);
// only this #version line differs (the port runs a 4.1 core context).
#version 410 core

uniform samplerCube u_texture;

in vec3 v_ray;

out vec4 fragColor;

void main(void)
{
    fragColor = texture(u_texture, normalize(v_ray));
}
