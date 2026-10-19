// Copy of Example13/shader/tessellation.cont.glsl (McNopper/OpenGL v3.2.7) with one port
// edit: u_tessLevel replaces the six hard-coded 4.0 levels, driven by the imgui slider.
#version 410 core

// Working with quads, so four vertices out
layout(vertices = 4) out;

// Tessellation level for all outer edges and both inner directions (C: 4.0).
uniform float u_tessLevel;

void main(void)
{
    // Outer tessellation level, so we have four "outer" rows and columns
    gl_TessLevelOuter[0] = u_tessLevel;
    gl_TessLevelOuter[1] = u_tessLevel;
    gl_TessLevelOuter[2] = u_tessLevel;
    gl_TessLevelOuter[3] = u_tessLevel;

    // Inner tessellation level, so we have four "inner" rows and columns
    gl_TessLevelInner[0] = u_tessLevel;
    gl_TessLevelInner[1] = u_tessLevel;

    // Finally, we have a four by four tessellated plane
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
