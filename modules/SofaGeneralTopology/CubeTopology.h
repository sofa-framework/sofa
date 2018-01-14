/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_CUBETOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_CUBETOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/MeshTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_GENERAL_TOPOLOGY_API CubeTopology : public MeshTopology
{
public:
    SOFA_CLASS(CubeTopology,MeshTopology);
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    CubeTopology(int nx, int ny, int nz);
    CubeTopology();
public:
    void setSize(int nx, int ny, int nz);

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    int getNx() const { return nx.getValue(); }
    int getNy() const { return ny.getValue(); }
    int getNz() const { return nz.getValue(); }

    void setNx(int n) { nx.setValue(n); setSize(); }
    void setNy(int n) { ny.setValue(n); setSize(); }
    void setNz(int n) { nz.setValue(n); setSize(); }

    virtual void init() override;
    virtual void reinit() override;

    enum Plane { PLANE_UNKNOWN=0,
            PLANE_X0,
            PLANE_X1,
            PLANE_Y0,
            PLANE_Y1,
            PLANE_Z0,
            PLANE_Z1
               };

    int point(int x, int y, int z, Plane p = PLANE_UNKNOWN) const;

    void setP0(const Vector3& val) { p0 = val; }
    void setDx(const Vector3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vector3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vector3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    void setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax);

    const Vector3& getP0() const { return p0; }
    const Vector3& getDx() const { return dx; }
    const Vector3& getDy() const { return dy; }
    const Vector3& getDz() const { return dz; }

    Vector3   getMin() const { return min.getValue();}
    Vector3   getMax() const { return max.getValue();}

    Vector3 getPoint(int i) const;
    virtual Vector3 getPoint(int x, int y, int z) const;
    bool hasPos()  const override { return true; }
    SReal getPX(int i)  const override { return getPoint(i)[0]; }
    SReal getPY(int i) const override { return getPoint(i)[1]; }
    SReal getPZ(int i) const override { return getPoint(i)[2]; }

    void setSplitNormals(bool b) {splitNormals.setValue(b);}

protected:
    Data<int> nx;
    Data<int> ny;
    Data<int> nz;
    Data<bool> internalPoints;
    Data<bool> splitNormals;

    Data< Vector3 > min, max;
    /// Position of point 0
    Vector3 p0;
    /// Distance between points in the grid. Must be perpendicular to each other
    Vector3 dx,dy,dz;
    SReal inv_dx2, inv_dy2, inv_dz2;

    virtual void setSize();
    void updatePoints();
    void updateEdges();
    void updateQuads();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
