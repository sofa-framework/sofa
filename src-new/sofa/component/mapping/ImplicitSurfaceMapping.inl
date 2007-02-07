#ifndef SOFA_COMPONENTS_IMPLICITSURFACEMAPPING_INL
#define SOFA_COMPONENTS_IMPLICITSURFACEMAPPING_INL

#include "ImplicitSurfaceMapping.h"

#include "Sofa-old/Core/Mapping.inl"
#include "Common/rmath.h"

#include <map>
#include <list>

namespace Sofa
{

namespace Components
{

using namespace Common;

template<class Real>
Real sqr(Real r)
{
    return r*r;
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::apply( OutVecCoord& out, const InVecCoord& in )
{
    InReal invStep = (InReal)(1/mStep);
    out.resize(0);
    clear();
    if (in.size()==0) return;
    InReal xmin, xmax;
    InReal ymin, ymax;
    xmin = xmax = in[0][0]*invStep;
    ymin = ymax = in[0][1]*invStep;
    const InReal r = (InReal)(getRadius() / mStep);
    std::map<int, std::list< InCoord > > sortParticles;
    for (unsigned int ip=0; ip<in.size(); ip++)
    {
        InCoord c0 = in[ip];
        if (c0[0] < mGridMin[0] || c0[0] > mGridMax[0] ||
            c0[1] < mGridMin[1] || c0[1] > mGridMax[1] ||
            c0[2] < mGridMin[2] || c0[2] > mGridMax[2])
            continue;
        InCoord c = c0 * invStep;
        if (c[0] < xmin)
            xmin = c[0];
        else if (c[0] > xmax)
            xmax = c[0];
        if (c[1] < ymin)
            ymin = c[1];
        else if (c[1] > ymax)
            ymax = c[1];
        int z0 = rceil(c[2]-r);
        int z1 = rfloor(c[2]+r);
        for (int z = z0; z < z1; ++z)
            sortParticles[z].push_back(c);
    }

    const int z0 = sortParticles.begin()->first - 1;
    const int nz = sortParticles.rbegin()->first - z0 + 2;
    const int y0 = rceil(ymin-r) - 1;
    const int ny = rfloor(ymax+r) - y0 + 2;
    const int x0 = rceil(xmin-r) - 1;
    const int nx = rfloor(xmax+r) - x0 + 2;

    planes.resize(2*nx*ny);
    P0 = planes.begin()+0;
    P1 = planes.begin()+nx*ny;

    //////// MARCHING CUBE ////////

    const OutReal isoval = (OutReal) getIsoValue();

    const int dx = 1;
    const int dy = nx;
    //const int dz = nx*ny;

    int x,y,z,i,mk;
    const int *tri;

    OutReal r2 = (OutReal)sqr(r);
    // First plane is all zero
    z = 0;
    newPlane();
    for (z=1; z<nz; z++)
    {
        newPlane();

        // Compute the data
        const std::list<InCoord>& particles = sortParticles[z0+z];
        for (typename std::list<InCoord>::const_iterator it = particles.begin(); it != particles.end(); ++it)
        {
            InCoord c = *it;
            int cx0 = rceil(c[0]-r);
            int cx1 = rfloor(c[0]+r);
            int cy0 = rceil(c[1]-r);
            int cy1 = rfloor(c[1]+r);
            OutCoord dp2;
            dp2[2] = (OutReal)sqr(z0+z-c[2]);
            i = (cx0-x0)+(cy0-y0)*nx;
            for (int y = cy0 ; y <= cy1 ; y++)
            {
                dp2[1] = (OutReal)sqr(y-c[1]);
                int ix = i;
                for (int x = cx0 ; x <= cx1 ; x++, ix++)
                {
                    dp2[0] = (OutReal)sqr(x-c[0]);
                    OutReal d2 = dp2[0]+dp2[1]+dp2[2];
                    if (d2 < r2)
                    {
                        // Soft object field function from the Wyvill brothers
                        // See http://astronomy.swin.edu.au/~pbourke/modelling/implicitsurf/
                        d2 /= r2;
                        (P1+ix)->data += (1 + (-4*d2*d2*d2 + 17*d2*d2 - 22*d2)/9);
                    }
                }
                i += nx;
            }
        }

        i=0;
        int edgecube[12];
        const int edgepts[12] = {0,1,0,1,0,1,0,1,2,2,2,2};
        typename std::vector<CubeData>::iterator base = planes.begin();
        int ip0 = P0-base;
        int ip1 = P1-base;
        edgecube[0]  = (ip0   -dy);
        edgecube[1]  = (ip0      );
        edgecube[2]  = (ip0      );
        edgecube[3]  = (ip0-dx   );
        edgecube[4]  = (ip1   -dy);
        edgecube[5]  = (ip1      );
        edgecube[6]  = (ip1      );
        edgecube[7]  = (ip1-dx   );
        edgecube[8]  = (ip1-dx-dy);
        edgecube[9]  = (ip1-dy   );
        edgecube[10] = (ip1      );
        edgecube[11] = (ip1-dx   );

        // First line is all zero
        {
            y=0;
            x=0;
            i+=nx;
        }
        for(y=1; y<ny; y++)
        {
            // First column is all zero
            x=0;
            ++i;

            for(x=1; x<nx; x++)
            {
                if (((P1+i)->data>isoval)^((P1+i-dx)->data>isoval))
                {
                    (P1+i)->p[0] = addPoint<0>(out, x0+x,y0+y,z0+z,(P1+i)->data,(P1+i-dx)->data,isoval);
                }
                if (((P1+i)->data>isoval)^((P1+i-dy)->data>isoval))
                {
                    (P1+i)->p[1] = addPoint<1>(out, x0+x,y0+y,z0+z,(P1+i)->data,(P1+i-dy)->data,isoval);
                }
                if (((P1+i)->data>isoval)^((P0+i)->data>isoval))
                {
                    (P1+i)->p[2] = addPoint<2>(out, x0+x,y0+y,z0+z,(P1+i)->data,(P0+i)->data,isoval);
                }

                // All points should now be created

                if ((P0+i-dx-dy)->data > isoval) mk = 1; else mk=0;
                if ((P0+i   -dy)->data > isoval) mk|= 2;
                if ((P0+i      )->data > isoval) mk|= 4;
                if ((P0+i-dx   )->data > isoval) mk|= 8;
                if ((P1+i-dx-dy)->data > isoval) mk|= 16;
                if ((P1+i   -dy)->data > isoval) mk|= 32;
                if ((P1+i      )->data > isoval) mk|= 64;
                if ((P1+i-dx   )->data > isoval) mk|= 128;

                tri=MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    typename std::vector<CubeData>::iterator b = base+i;
                    if (addFace((b+edgecube[tri[0]])->p[edgepts[tri[0]]],
                            (b+edgecube[tri[1]])->p[edgepts[tri[1]]],
                            (b+edgecube[tri[2]])->p[edgepts[tri[2]]], out.size())<0)
                    {
                        std::cerr << "  mk=0x"<<std::hex<<mk<<std::dec<<" p1="<<tri[0]<<" p2="<<tri[1]<<" p3="<<tri[2]<<std::endl;
                        for (int e=0; e<12; e++) std::cerr << "  e"<<e<<"="<<(b+edgecube[e])->p[edgepts[e]];
                        std::cerr<<std::endl;
                    }
                    tri+=3;
                }
                ++i;
            }
        }
    }
    std::cout << out.size() << " points, "<<seqTriangles.getValue().size()<<" faces."<<std::endl;
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::newPlane()
{
    CubeData c;
    c.p[0] = -1;
    c.p[1] = -1;
    c.p[2] = -1;
    c.data = 0;
    typename std::vector<CubeData>::iterator P = P0;
    P0 = P1;
    P1 = P;
    int n = planes.size()/2;
    for (int i=0; i<n; ++i,++P)
        *P = c;
    //plane0.swap(plane1);
    //plane1.fill(c);
}


template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::applyJ( OutVecDeriv& /*out*/, const InVecDeriv& /*in*/ )
{
}

} // namespace Components

} // namespace Sofa

#endif
