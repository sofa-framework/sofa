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

void MarchingCube::generateSurfaceMesh(const double isoval, const double mstep, const double invStep,
                                       const Vec3d& gridmin, const Vec3d& gridmax,
                                       std::function<void(std::vector<Vec3d>&, std::vector<double>&)> getFieldValueAt,
                                       SeqCoord& tmpPoints, SeqTriangles& tmpTriangles)
{    
    int nx = floor((gridmax.x() - gridmin.x()) * invStep) + 1 ;
    int ny = floor((gridmax.y() - gridmin.y()) * invStep) + 1 ;
    int nz = floor((gridmax.z() - gridmin.z()) * invStep) + 1 ;

    // Marching cubes only works for a grid size larger than two
    if( nz < 2 || ny < 2 || nx < 2 )
        return;

    double cx,cy,cz;
    int z,mk;
    const int *tri;

    // Creates two planes
    CubeData c{{-1,-1,-1},0};
    planes.resize(2*nx*ny);
    for(size_t i=0;i<planes.size();++i)
    {
        planes[i] = c;
    }
    // Keep two pointers for the first plane and the seconds
    P0 = planes.begin();
    P1 = planes.begin()+nx*ny;

    const int dx = 1;
    const int dy = nx;

    z = 0;

    auto fillPlane = [getFieldValueAt](std::vector<Vec3d> &positions, std::vector<double>& output,
                           double mstep, double gridmin_y, double gridmin_x, int ny, int nx, float cz,
                           std::vector<CubeData>::iterator itDestPlane)
    {
        for (int i=0, y = 0 ; y < ny ; ++y)
        {
            double cy = gridmin_y + mstep * y ;
            for (int x = 0 ; x < nx ; ++x)
            {
                double cx = gridmin_x + mstep * x ;
                positions[i++].set(cx, cy, cz );
            }
        }
        getFieldValueAt(positions, output) ;

        for(auto res : output){
            itDestPlane->data = res;
            itDestPlane++;
        }
    };

    std::vector<Vec3d> positions;
    std::vector<double> output;
    positions.resize(nx*ny);
    output.resize(nx*ny);

    fillPlane(positions, output, mstep, gridmin.y(), gridmin.x(), ny, nx, gridmin.z(), P0);
    for (z=1; z<=nz; ++z)
    {
        fillPlane(positions, output, mstep, gridmin.y(), gridmin.x(), ny, nx, gridmin.z() + mstep * z, P1);

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

        unsigned int di = nx;
        for(int y=1; y<ny; y++)
        {
            // First column is all zero
            int x=0;
            ++di;
            for(x=1; x<nx; x++)
            {
                Vec3d pos(x, y, z);
                if (((P1+di)->data>isoval)^((P1+di-dx)->data>isoval))
                {
                    (P1+di)->p[0] = addPoint(tmpPoints, 0, pos,gridmin, (P1+di)->data,(P1+di-dx)->data, mstep, isoval);
                }
                if (((P1+di)->data>isoval)^((P1+di-dy)->data>isoval))
                {
                    (P1+di)->p[1] = addPoint(tmpPoints, 1, pos,gridmin,(P1+di)->data,(P1+di-dy)->data, mstep, isoval);
                }
                if (((P1+di)->data>isoval)^((P0+di)->data>isoval))
                {
                    (P1+di)->p[2] = addPoint(tmpPoints, 2, pos,gridmin,(P1+di)->data,(P0+di)->data, mstep, isoval);
                }

                // All points should now be created
                if ((P0+di-dx-dy)->data > isoval) mk = 1;
                else mk=0;
                if ((P0+di   -dy)->data > isoval) mk|= 2;
                if ((P0+di      )->data > isoval) mk|= 4;
                if ((P0+di-dx   )->data > isoval) mk|= 8;
                if ((P1+di-dx-dy)->data > isoval) mk|= 16;
                if ((P1+di   -dy)->data > isoval) mk|= 32;
                if ((P1+di      )->data > isoval) mk|= 64;
                if ((P1+di-dx   )->data > isoval) mk|= 128;

                tri=sofa::helper::MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    typename std::vector<CubeData>::iterator b = base+di;
                    addFace(tmpTriangles,
                                (b+edgecube[tri[0]])->p[edgepts[tri[0]]],
                                (b+edgecube[tri[1]])->p[edgepts[tri[1]]],
                                (b+edgecube[tri[2]])->p[edgepts[tri[2]]], tmpPoints.size());
                    tri+=3;
                }
                ++di;
            }
        }
        std::swap(P0, P1);
    }
}

}
