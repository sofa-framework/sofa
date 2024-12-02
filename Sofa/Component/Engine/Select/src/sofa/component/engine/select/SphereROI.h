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

#include <sofa/core/objectmodel/RenamedData.h>

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
    using Real = Real_t<DataTypes>;
    using typename Inherit::CPos;
    using typename Inherit::PointID;
    using typename Inherit::Edge;
    using typename Inherit::Triangle;

protected:
    SphereROI();
    ~SphereROI() override = default;

public:
    bool roiDoUpdate() override;
    void roiDraw(const core::visual::VisualParams* vparams) override;
    void roiComputeBBox(const core::ExecParams* params, type::BoundingBox& bbox) override;

protected:
    bool testEdgeAngle(const Edge& e) const;
    bool testTriangleAngle(const Triangle& t) const;

    bool isPointInSphere(const CPos& c, const Real& r, const CPos& p) const;

    bool isPointInROI(const CPos& p) const override;
    bool isEdgeInROI(const Edge& e) const override;
    bool isEdgeInStrictROI(const Edge& e) const override;
    bool isTriangleInROI(const Triangle& t) const override;
    bool isTriangleInStrictROI(const Triangle& t) const override;

public:
    //Input
    Data< type::vector<CPos> > d_centers; ///< Center(s) of the sphere(s)
    Data< type::vector<Real> > d_radii; ///< Radius(i) of the sphere(s)
    
    Data< type::Vec3 > d_direction; ///< Edge direction(if edgeAngle > 0)
    Data< type::Vec3 > d_normal; ///< Normal direction of the triangles (if triAngle > 0)
    Data< Real > d_edgeAngle; ///< Max angle between the direction of the selected edges and the specified direction
    Data< Real > d_triAngle; ///< Max angle between the normal of the selected triangle and the specified normal direction

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< type::vector<CPos> > centers;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< type::vector<Real> > radii;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< type::Vec3 > direction;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< type::Vec3 > normal;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< Real > edgeAngle;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData< Real > triAngle;
};

#if !defined(SOFA_COMPONENT_ENGINE_SPHEREROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::engine::select
