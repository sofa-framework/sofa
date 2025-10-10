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
#pragma once
#include <SofaImplicitField/config.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaImplicitField/components/geometry/ScalarField.h>
#include <sofa/type/Vec.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace sofaimplicitfield
{

typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef sofa::type::vector<sofa::type::Vec3d> SeqCoord;
using sofa::type::Vec3d;

class MarchingCube
{
public:
    void generateSurfaceMesh(const double isoval, const double mstep, const double invStep,
                             const Vec3d& gridmin, const Vec3d& gridmax,
                             std::function<void (std::vector<Vec3d> &, std::vector<double> &)> field,
                             SeqCoord& tmpPoints, SeqTriangles& tmpTriangles);

private:
    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
        double data;
    };

    sofa::type::vector<CubeData> planes;
    typename sofa::type::vector<CubeData>::iterator P0; /// Pointer to first plane
    typename sofa::type::vector<CubeData>::iterator P1; /// Pointer to second plane

    int addPoint(SeqCoord& v, int i, Vec3d pos, const Vec3d& gridmin, double v0, double v1, double step, double iso)
    {
        pos[i] -= (iso-v0)/(v1-v0);
        v.push_back( (pos * step)+gridmin ) ;
        return v.size()-1;
    }

    int addFace(SeqTriangles& triangles, int p1, int p2, int p3, int nbp)
    {
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
            triangles.push_back(Triangle(p1, p3, p2));
            return triangles.size()-1;
        }
        else
        {
            return -1;
        }
    }
};


}

