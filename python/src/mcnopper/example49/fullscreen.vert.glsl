// Verbatim copy of Example49/shader/fullscreen.vert.glsl (McNopper/OpenGL v3.2.7);
// only this #version line differs (the port runs a 4.1 core context).
#version 410 core

// Generates a fullscreen quad from gl_VertexID — no VBO needed.
// gl_VertexID: 0=(−1,−1), 1=(1,−1), 2=(−1,1), 3=(1,1)

out vec2 v_texCoord;

void main(void)
{
    vec2 pos    = vec2((gl_VertexID & 1) * 2 - 1, (gl_VertexID >> 1) * 2 - 1);
    v_texCoord  = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
