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
#include <sofa/component/engine/select/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/engine/select/BaseROI.h>

namespace sofa::component::engine::select
{

/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given sphere.
 */
template <class DataTypes>
class SphereROI : public BaseROI<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SphereROI, DataTypes), SOFA_TEMPLATE(BaseROI, DataTypes));
    using Inherit = BaseROI<DataTypes>;

protected:
    SphereROI();
    ~SphereROI() override = default;

public:
    void roiInit() override;
    bool roiDoUpdate() override;
    void roiDraw(const core::visual::VisualParams* vparams) override;

protected:
    bool isPointInSphere(const CPos& c, const Real& r, const CPos& p);
    bool isPointInSphere(const PointID& pid, const Real& r, const CPos& p);
    bool isEdgeInSphere(const CPos& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Edge& edge);
    bool isTriangleInSphere(const CPos& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Triangle& triangle);
    bool isQuadInSphere(const CPos& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Quad& quad);
    bool isTetrahedronInSphere(const CPos& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Tetra& tetrahedron);

    bool isPointInROI(const CPos& p) override;
    bool isEdgeInROI(const Edge& e) override;
    bool isEdgeInStrictROI(const Edge& e) override;
    bool isTriangleInROI(const Triangle& t) override;
    bool isTriangleInStrictROI(const Triangle& t) override;

public:
    //Input
    Data< type::vector<CPos> > d_centers; ///< Center(s) of the sphere(s)
    Data< type::vector<Real> > d_radii; ///< Radius(i) of the sphere(s)
    
    Data< type::Vec3 > d_direction; ///< Edge direction(if edgeAngle > 0)
    Data< type::Vec3 > d_normal; ///< Normal direction of the triangles (if triAngle > 0)
    Data< Real > d_edgeAngle; ///< Max angle between the direction of the selected edges and the specified direction
    Data< Real > d_triAngle; ///< Max angle between the normal of the selected triangle and the specified normal direction

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< type::vector<CPos> > centers;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< type::vector<Real> > radii;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< type::Vec3 > direction;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< type::Vec3 > normal;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< Real > edgeAngle;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    Data< Real > triAngle;
};

#if !defined(SOFA_COMPONENT_ENGINE_SPHEREROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::engine::select
