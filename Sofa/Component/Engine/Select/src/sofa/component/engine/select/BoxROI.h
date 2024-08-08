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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/engine/select/BaseROI.h>

namespace sofa::component::engine::select::boxroi
{

/// This namespace is used to avoid the leaking of the 'using' on includes.
/// BoxROI is defined in namespace in sofa::component::engine::boxroi:BoxROI
/// It is then import into sofa::component::engine::BoxROI to not break the
/// API.

    using core::objectmodel::BaseObjectDescription ;
    using sofa::core::behavior::MechanicalState ;
    using core::topology::BaseMeshTopology ;
    using core::behavior::MechanicalState ;
    using core::objectmodel::BaseContext ;
    using core::objectmodel::BaseObject ;
    using core::visual::VisualParams ;
    using core::objectmodel::Event ;
    using core::ExecParams ;
    using core::DataEngine ;
    using std::string ;



/**
 * This class find all the points/edges/triangles/quads/tetrahedras/hexahedras located inside given boxes.
 */
template <class DataTypes>
class BoxROI : public BaseROI<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxROI,DataTypes), SOFA_TEMPLATE(BaseROI, DataTypes));
    using Inherit = BaseROI<DataTypes>;

    typedef type::Vec<10, SReal> Vec10;
    using Real = Real_t<DataTypes>;
    using typename Inherit::CPos;

public:
    void roiInit() override;
    bool roiDoUpdate() override;
    void roiDraw(const VisualParams* vparams) override;
    void roiComputeBBox(const ExecParams* params, type::BoundingBox& bbox) override;

public:
    //Input
    Data<type::vector<type::Vec6> >  d_alignedBoxes; ///< List of boxes, each defined by two 3D points : xmin,ymin,zmin, xmax,ymax,zmax
    Data<type::vector<Vec10> > d_orientedBoxes; ///< each box is defined using three point coordinates and a depth value
   
protected:

    struct OrientedBox
    {
        typedef typename DataTypes::Real Real;
        typedef type::Vec<3,Real> Vec3;

        Vec3 p0, p2;
        Vec3 normal;
        Vec3 plane0, plane1, plane2, plane3;
        double width, length, depth;
    };

    type::vector<OrientedBox> m_orientedBoxes;

    BoxROI();
    ~BoxROI() override = default;

    void computeOrientedBoxes();

    bool isPointInOrientedBox(const CPos& p, const OrientedBox& box) const;
    static bool isPointInAlignedBox(const typename DataTypes::CPos& p, const type::Vec6& box);
    void getPointsFromOrientedBox(const Vec10& box, type::vector<type::Vec3> &points) const;

    bool isPointInROI(const CPos& p) const override;
};

#if !defined(SOFA_COMPONENT_ENGINE_BOXROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<defaulttype::Vec6Types>;

#endif

} // namespace sofa::component::engine::select::boxroi

namespace sofa::component::engine::select
{

/// Import sofa::component::engine::boxroi::BoxROI into
/// into the sofa::component::engine namespace.
using boxroi::BoxROI ;

} // namespace sofa::component::engine::select
