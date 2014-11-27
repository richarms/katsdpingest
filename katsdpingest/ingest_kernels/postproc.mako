<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void postproc(
    GLOBAL float2 * RESTRICT vis,
    GLOBAL float * RESTRICT weights,
    const GLOBAL unsigned char * RESTRICT flags,
    GLOBAL float2 * RESTRICT cont_vis,
    GLOBAL float * RESTRICT cont_weights,
    GLOBAL unsigned char * RESTRICT cont_flags,
    int stride)
{
    const int C = ${cont_factor};
    int baseline = get_global_id(0);
    int cont_channel = get_global_id(1);
    int channel0 = cont_channel * C;

    float2 cv;
    cv.x = 0.0f;
    cv.y = 0.0f;
    float cw = 0.0f;
    unsigned char cf = 0xff;
    for (int i = 0; i < C; i++)
    {
        int channel = channel0 + i;
        int addr = channel * stride + baseline;
        GLOBAL float2 *vptr = &vis[addr];
        float2 v = *vptr;
        GLOBAL float *wptr = &weights[addr];
        float w = *wptr;
        unsigned char f = flags[addr];
        cv.x += v.x;
        cv.y += v.y;
        cw += w;
        cf &= f;
        float scale = 1.0f / w;
        if (f)
            *wptr = 0.0f;
        v.x *= scale;
        v.y *= scale;
        *vptr = v;
    }

    float scale = 1.0 / cw;
    cv.x *= scale;
    cv.y *= scale;
    if (cf)
        cw = 0.0f;
    int cont_addr = cont_channel * stride + baseline;
    cont_vis[cont_addr] = cv;
    cont_weights[cont_addr] = cw;
    cont_flags[cont_addr] = cf;
}
