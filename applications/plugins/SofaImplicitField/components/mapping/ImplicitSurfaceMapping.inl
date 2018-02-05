/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_INL

#include "ImplicitSurfaceMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/rmath.h>
#include <map>
#include <list>



namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::init()
{
    core::Mapping<In,Out>::init();
    topology::MeshTopology::init();
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->Inherit::parse(arg);
    if ( arg->getAttribute("minx") || arg->getAttribute("miny") || arg->getAttribute("minz"))
        this->setGridMin(arg->getAttributeAsFloat("minx",-100.0),
                         arg->getAttributeAsFloat("miny",-100.0),
                         arg->getAttributeAsFloat("minz",-100.0));
    if (arg->getAttribute("maxx") || arg->getAttribute("maxy") || arg->getAttribute("maxz"))
        this->setGridMax(arg->getAttributeAsFloat("maxx",100.0),
                         arg->getAttributeAsFloat("maxy",100.0),
                         arg->getAttributeAsFloat("maxz",100.0));
}

template<class Real>
Real sqr(Real r)
{
    return r*r;
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    OutVecCoord &out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    InReal invStep = (InReal)(1/mStep.getValue());
    out.resize(0);
    clear();

    if (in.size()==0)
    {
        dOut.endEdit();
        return;
    }

    InReal xmin, xmax;
    InReal ymin, ymax;
    xmin = xmax = in[0][0]*invStep;
    ymin = ymax = in[0][1]*invStep;
    const InReal r = (InReal)(getRadius() / mStep.getValue());
    std::map<int, std::list< InCoord > > sortParticles;
    for (unsigned int ip=0; ip<in.size(); ip++)
    {
        InCoord c0 = in[ip];
        if (c0[0] < (*mGridMin.beginEdit())[0] || c0[0] > (*mGridMax.beginEdit())[0] ||
            c0[1] < (*mGridMin.beginEdit())[1] || c0[1] > (*mGridMax.beginEdit())[1] ||
            c0[2] < (*mGridMin.beginEdit())[2] || c0[2] > (*mGridMax.beginEdit())[2])
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
        int z0 = helper::rceil(c[2]-r);
        int z1 = helper::rfloor(c[2]+r);
        for (int z = z0; z < z1; ++z)
            sortParticles[z].push_back(c);
    }

    const int z0 = sortParticles.begin()->first - 1;
    const int nz = sortParticles.rbegin()->first - z0 + 2;
    const int y0 = helper::rceil(ymin-r) - 1;
    const int ny = helper::rfloor(ymax+r) - y0 + 2;
    const int x0 = helper::rceil(xmin-r) - 1;
    const int nx = helper::rfloor(xmax+r) - x0 + 2;

    (*planes.beginEdit()).resize(2*nx*ny);
    P0 = (*planes.beginEdit()).begin()+0;
    P1 = (*planes.beginEdit()).begin()+nx*ny;

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
            int cx0 = helper::rceil(c[0]-r);
            int cx1 = helper::rfloor(c[0]+r);
            int cy0 = helper::rceil(c[1]-r);
            int cy1 = helper::rfloor(c[1]+r);
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
        typename std::vector<CubeData>::iterator base = (*planes.beginEdit()).begin();
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

                tri=sofa::helper::MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    typename std::vector<CubeData>::iterator b = base+i;
                    if (addFace((b+edgecube[tri[0]])->p[edgepts[tri[0]]],
                            (b+edgecube[tri[1]])->p[edgepts[tri[1]]],
                            (b+edgecube[tri[2]])->p[edgepts[tri[2]]], out.size())<0)
                    {
                        serr << "  mk=0x"<<std::hex<<mk<<std::dec<<" p1="<<tri[0]<<" p2="<<tri[1]<<" p3="<<tri[2]<<sendl;
                        for (int e=0; e<12; e++) serr << "  e"<<e<<"="<<(b+edgecube[e])->p[edgepts[e]];
                        serr<<sendl;
                    }
                    tri+=3;
                }
                ++i;
            }
        }
    }

    dOut.endEdit();
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
    int n = planes.getValue().size()/2;
    for (int i=0; i<n; ++i,++P)
        *P = c;
    //plane0.swap(plane1);
    //plane1.fill(c);
}


template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& /*dOut*/, const Data<InVecDeriv>& /*dIn*/)
{
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
