// Orthographic back-to-front alpha compositing of projected Gaussian ellipses.
#include <algorithm>
#include <cmath>
extern "C" void render_gaussians(const float* g, int count, int size, float* rgb, float* opacity) {
    for (int n=0;n<count;++n) {
        const float* p=g+11*n;
        int x0=std::max(0,(int)std::floor(p[0]-p[5])),x1=std::min(size-1,(int)std::ceil(p[0]+p[5]));
        int y0=std::max(0,(int)std::floor(p[1]-p[6])),y1=std::min(size-1,(int)std::ceil(p[1]+p[6]));
        for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
            float dx=x-p[0],dy=y-p[1];float power=-.5f*(p[2]*dx*dx+2*p[3]*dx*dy+p[4]*dy*dy);
            if (power< -4.5f) continue;
            float alpha=std::min(.99f,p[7]*std::exp(power));
            if (alpha<1.f/255.f) continue;
            int k=y*size+x;
            for (int c=0;c<3;++c) rgb[3*k+c]=alpha*p[8+c]+(1-alpha)*rgb[3*k+c];
            opacity[k]=alpha+(1-alpha)*opacity[k];
        }
    }
}
