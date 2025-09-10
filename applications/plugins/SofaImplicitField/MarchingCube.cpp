/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2025 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaImplicitField/config.h>

#include <SofaImplicitField/MarchingCube.h>
#include <sofa/helper/MarchingCubeUtility.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaImplicitField/components/geometry/ScalarField.h>
#include <sofa/type/Vec.h>

namespace sofaimplicitfield
{

void MarchingCube::newPlane()
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
}

void MarchingCube::generateSurfaceMesh(const double isoval, const double mstep, const double invStep,
                                       const Vec3d& gridmin, const Vec3d& gridmax,
                                       std::function<void(std::vector<Vec3d>&, std::vector<double>&)> getFieldValueAt,
                                       SeqCoord& tmpPoints, SeqTriangles& tmpTriangles)
{    
    int nx = floor((gridmax.x() - gridmin.x()) * invStep) + 1 ;
    int ny = floor((gridmax.y() - gridmin.y()) * invStep) + 1 ;
    int nz = floor((gridmax.z() - gridmin.z()) * invStep) + 1 ;

    if( nz <= 0 || ny <= 0 || nx <= 0 )
        return;

    double cx,cy,cz;
    int x,y,z,i,mk;
    const int *tri;

    planes.resize(2*(nx)*(ny));
    P0 = planes.begin()+0;
    P1 = planes.begin()+nx*ny;

    const int dx = 1;
    const int dy = nx;

    z = 0;
    newPlane();

    i = 0 ;
    cz = gridmin.z()  ;

    std::vector<Vec3d> positions;
    std::vector<double> output;
    positions.resize(ny*nx);
    output.resize(nx*ny);
    for (int y = 0 ; y < ny ; ++y)
    {
        cy = gridmin.y() + mstep * y ;
        for (int x = 0 ; x < nx ; ++x, ++i)
        {
            cx = gridmin.x() + mstep * x ;
            positions[i].set(cx, cy, cz );
        }
    }
    getFieldValueAt(positions, output) ;

    // Copy back the data into planes.
    auto it = P1;
    for(auto res : output){
        it->data = res;
        it++;
    }

    for (z=1; z<=nz; ++z)
    {
        newPlane();

//        i = 0 ;
        cz = gridmin.z() + mstep * z ;
        //positions.clear();
//        for (int y = 0 ; y < ny ; ++y)
//        {
//            cy = gridmin.y() + mstep * y ;
//            for (int x = 0 ; x < nx ; ++x, ++i)
//            {
//                cx = gridmin.x() + mstep * x ;
//                //positions[i].set(cx, cy, cz);
//            }
//        }

        positions[0].z() = cz;
        getFieldValueAt(positions, output) ;

        // Copy back the data into planes.
        auto it = P1;
        for(auto res : output)
        {
            it->data = res;
            it++;
        }

        unsigned int i=0;
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
                Vec3d pos(x, y, z);
                if (((P1+i)->data>isoval)^((P1+i-dx)->data>isoval))
                {
                    (P1+i)->p[0] = addPoint(tmpPoints, 0, pos,gridmin, (P1+i)->data,(P1+i-dx)->data, mstep, isoval);
                }
                if (((P1+i)->data>isoval)^((P1+i-dy)->data>isoval))
                {
                    (P1+i)->p[1] = addPoint(tmpPoints, 1, pos,gridmin,(P1+i)->data,(P1+i-dy)->data, mstep, isoval);
                }
                if (((P1+i)->data>isoval)^((P0+i)->data>isoval))
                {
                    (P1+i)->p[2] = addPoint(tmpPoints, 2, pos,gridmin,(P1+i)->data,(P0+i)->data, mstep, isoval);
                }

                // All points should now be created
                if ((P0+i-dx-dy)->data > isoval) mk = 1;
                else mk=0;
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
                    addFace(tmpTriangles,
                                (b+edgecube[tri[0]])->p[edgepts[tri[0]]],
                                (b+edgecube[tri[1]])->p[edgepts[tri[1]]],
                                (b+edgecube[tri[2]])->p[edgepts[tri[2]]], tmpPoints.size());
                    tri+=3;
                }
                ++i;
            }
        }
    }
}

}
