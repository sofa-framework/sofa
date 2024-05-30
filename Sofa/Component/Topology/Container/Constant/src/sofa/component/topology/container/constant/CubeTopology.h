/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/topology/container/constant/config.h>

#include <sofa/component/topology/container/constant/MeshTopology.h>

namespace sofa::component::topology::container::constant
{

namespace
{
    using sofa::type::Vec3;
}

class SOFA_COMPONENT_TOPOLOGY_CONTAINER_CONSTANT_API CubeTopology : public MeshTopology
{
public:
    SOFA_CLASS(CubeTopology,MeshTopology);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

protected:
    CubeTopology(int nx, int ny, int nz);
    CubeTopology();
public:
    void setSize(int nx, int ny, int nz);

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    int getNx() const { return d_nx.getValue(); }
    int getNy() const { return d_ny.getValue(); }
    int getNz() const { return d_nz.getValue(); }

    void setNx(int n) { d_nx.setValue(n); setSize(); }
    void setNy(int n) { d_ny.setValue(n); setSize(); }
    void setNz(int n) { d_nz.setValue(n); setSize(); }

    void init() override;
    void reinit() override;

    enum Plane { PLANE_UNKNOWN=0,
            PLANE_X0,
            PLANE_X1,
            PLANE_Y0,
            PLANE_Y1,
            PLANE_Z0,
            PLANE_Z1
               };

    int point(int x, int y, int z, Plane p = PLANE_UNKNOWN) const;

    void setP0(const Vec3& val) { p0 = val; }
    void setDx(const Vec3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vec3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vec3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    void setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax);

    const Vec3& getP0() const { return p0; }
    const Vec3& getDx() const { return dx; }
    const Vec3& getDy() const { return dy; }
    const Vec3& getDz() const { return dz; }

    Vec3   getMin() const { return d_min.getValue();}
    Vec3   getMax() const { return d_max.getValue();}

    Vec3 getPoint(int i) const;
    virtual Vec3 getPoint(int x, int y, int z) const;
    bool hasPos()  const override { return true; }
    SReal getPX(Index i)  const override { return getPoint(i)[0]; }
    SReal getPY(Index i) const override { return getPoint(i)[1]; }
    SReal getPZ(Index i) const override { return getPoint(i)[2]; }

    void setSplitNormals(bool b) {d_splitNormals.setValue(b);}

protected:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<int> nx;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<int> ny;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<int> nz;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<bool> internalPoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<bool> splitNormals;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<Vec3> min;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data<Vec3> max;


    Data<int> d_nx; ///< z grid resolution
    Data<int> d_ny;
    Data<int> d_nz;
    Data<bool> d_internalPoints; ///< include internal points (allow a one-to-one mapping between points from RegularGridTopology and CubeTopology)
    Data<bool> d_splitNormals; ///< split corner points to have planar normals

    Data< Vec3 > d_min; ///< Min
    Data< Vec3 > d_max; ///< Max
    /// Position of point 0
    Vec3 p0;
    /// Distance between points in the grid. Must be perpendicular to each other
    Vec3 dx,dy,dz;
    SReal inv_dx2, inv_dy2, inv_dz2;

    virtual void setSize();
    void updatePoints();
    void updateEdges();
    void updateQuads();
};

} // namespace sofa::component::topology::container::constant
