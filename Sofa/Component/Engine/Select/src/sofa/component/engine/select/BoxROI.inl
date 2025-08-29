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
#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <limits>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/helper/accessor.h>

#include <sofa/component/engine/select/BaseROI.inl>

namespace sofa::component::engine::select::boxroi
{

using core::behavior::BaseMechanicalState ;
using core::topology::TopologyContainer ;
using core::topology::BaseMeshTopology ;
using core::objectmodel::ComponentState ;
using core::objectmodel::BaseData ;
using core::objectmodel::Event ;
using core::ExecParams ;
using type::Vec3 ;
using type::Vec4f ;
using helper::WriteOnlyAccessor ;
using helper::ReadAccessor ;

template <class DataTypes>
BoxROI<DataTypes>::BoxROI()
    : Inherit()
    , d_alignedBoxes( initData(&d_alignedBoxes, "box", "List of boxes, each defined by two 3D points : xmin,ymin,zmin, xmax,ymax,zmax") )
    , d_orientedBoxes( initData(&d_orientedBoxes, "orientedBox", "List of boxes defined by 3 points (p0, p1, p2) and a depth distance \n"
                                "A parallelogram will be defined by (p0, p1, p2, p3 = p0 + (p2-p1)). \n"
                                "The box will finaly correspond to the parallelogram extrusion of depth/2 \n"
                                "along its normal and depth/2 in the opposite direction. ") )

    /// In case you add a new attribute please also add it into to the BoxROI_test.cpp::attributesTests
    /// In case you want to remove or rename an attribute, please keep it as-is but add a warning message
    /// using msg_warning saying to the user of this component that the attribute is deprecated and solutions/replacement
    /// he has to fix his scene.
{
    sofa::helper::getWriteOnlyAccessor(this->d_indices).push_back(0);

    this->addInput(&d_alignedBoxes);
    this->addInput(&d_orientedBoxes);

    this->addAlias(&this->d_drawROI, "drawBoxes");
    this->addAlias(&this->d_quads, "quad");
    this->addAlias(&this->d_computeQuads, "computeQuad");
    this->addAlias(&this->d_quadsInROI, "quadInROI");
}

template <class DataTypes>
void BoxROI<DataTypes>::roiInit()
{
    if(!d_alignedBoxes.isSet() && !d_orientedBoxes.isSet())
    {
        auto alignedBoxes = sofa::helper::getWriteOnlyAccessor(d_alignedBoxes);
        alignedBoxes.push_back(type::Vec6(0,0,0,1,1,1));
    }

    auto alignedBoxes = sofa::helper::getWriteOnlyAccessor(d_alignedBoxes);
    if (!alignedBoxes.empty())
    {
        for (unsigned int bi=0; bi<alignedBoxes.size(); ++bi)
        {
            if (alignedBoxes[bi][0] > alignedBoxes[bi][3]) std::swap(alignedBoxes[bi][0], alignedBoxes[bi][3]);
            if (alignedBoxes[bi][1] > alignedBoxes[bi][4]) std::swap(alignedBoxes[bi][1], alignedBoxes[bi][4]);
            if (alignedBoxes[bi][2] > alignedBoxes[bi][5]) std::swap(alignedBoxes[bi][2], alignedBoxes[bi][5]);
        }
    }

    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        static const std::string message = "\nOriented bounding boxes are not supported in " + std::to_string(DataTypes::spatial_dimensions) + "D";
        d_orientedBoxes.setHelp(d_orientedBoxes.getHelp() + message);
        msg_warning_when(d_orientedBoxes.isSet()) << message;
    }

    computeOrientedBoxes();
}

template <class DataTypes>
void BoxROI<DataTypes>::computeOrientedBoxes()
{
    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        return;
    }

    const vector<Vec10>& orientedBoxes = d_orientedBoxes.getValue();

    if(orientedBoxes.empty())
        return;

    m_orientedBoxes.resize(orientedBoxes.size());

    for(unsigned int i=0; i<orientedBoxes.size(); i++)
    {
        const Vec10& box = orientedBoxes[i];

        const type::Vec3 p0 = type::Vec3(box[0], box[1], box[2]);
        const type::Vec3 p1 = type::Vec3(box[3], box[4], box[5]);
        const type::Vec3 p2 = type::Vec3(box[6], box[7], box[8]);
        double depth = box[9];

        type::Vec3 normal = (p1-p0).cross(p2-p0);
        normal.normalize();

        const type::Vec3 p3 = p0 + (p2-p1);
        const type::Vec3 p6 = p2 + normal * depth;

        type::Vec3 plane0 = (p1-p0).cross(normal);
        plane0.normalize();

        type::Vec3 plane1 = (p2-p3).cross(p6-p3);
        plane1.normalize();

        type::Vec3 plane2 = (p3-p0).cross(normal);
        plane2.normalize();

        type::Vec3 plane3 = (p2-p1).cross(p6-p2);
        plane3.normalize();


        m_orientedBoxes[i].p0 = p0;
        m_orientedBoxes[i].p2 = p2;
        m_orientedBoxes[i].normal = normal;
        m_orientedBoxes[i].plane0 = plane0;
        m_orientedBoxes[i].plane1 = plane1;
        m_orientedBoxes[i].plane2 = plane2;
        m_orientedBoxes[i].plane3 = plane3;
        m_orientedBoxes[i].width = fabs(dot((p2-p0),plane0));
        m_orientedBoxes[i].length = fabs(dot((p2-p0),plane2));
        m_orientedBoxes[i].depth = depth;
    }
}


template <class DataTypes>
bool BoxROI<DataTypes>::isPointInOrientedBox(const CPos& point, const OrientedBox& box) const
{
    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        return false;
    }
    else
    {
        const type::Vec3 pv0 = type::Vec3(point[0]-box.p0[0], point[1]-box.p0[1], point[2]-box.p0[2]);
        const type::Vec3 pv1 = type::Vec3(point[0]-box.p2[0], point[1]-box.p2[1], point[2]-box.p2[2]);

        if( fabs(dot(pv0, box.plane0)) <= box.width && fabs(dot(pv1, box.plane1)) <= box.width )
        {
            if ( fabs(dot(pv0, box.plane2)) <= box.length && fabs(dot(pv1, box.plane3)) <= box.length )
            {
                if ( !(fabs(dot(pv0, box.normal)) <= fabs(box.depth/2)) )
                    return false;
            }
            else
                return false;
        }
        else
            return false;

        return true;
    }
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInAlignedBox(const typename DataTypes::CPos& p, const type::Vec6& box)
{
    static_assert(std::is_same_v<typename DataTypes::CPos::size_type, typename type::Vec6::size_type>);

    for (typename type::Vec6::size_type i = 0; i < DataTypes::spatial_dimensions; ++i)
    {
        if (p[i] < box[i] || p[i] > box[i + 3])
        {
            return false;
        }
    }
    return true;
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInROI(const CPos& p) const
{
    const vector<type::Vec6>& alignedBoxes = d_alignedBoxes.getValue();

    for (unsigned int i=0; i<alignedBoxes.size(); ++i)
        if (isPointInAlignedBox(p, alignedBoxes[i]))
            return true;

    if constexpr (DataTypes::spatial_dimensions == 3)
    {
        for (unsigned int i=0; i<m_orientedBoxes.size(); ++i)
            if (isPointInOrientedBox(p, m_orientedBoxes[i]))
                return true;
    }

    return false;
}

template <class DataTypes>
bool BoxROI<DataTypes>::roiDoUpdate()
{
    return !d_alignedBoxes.getValue().empty() || !d_orientedBoxes.getValue().empty();
}

template <class DataTypes>
void BoxROI<DataTypes>::roiDraw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->setLightingEnabled(false);
    const float linesWidth = std::max(this->d_drawSize.getValue(), 1.0f);

    const vector<type::Vec6>&  alignedBoxes =d_alignedBoxes.getValue();
    const vector<Vec10>& orientedBoxes=d_orientedBoxes.getValue();

    constexpr const sofa::type::RGBAColor& color = sofa::type::RGBAColor::cyan();

    vparams->drawTool()->setMaterial(color);
    for (const auto& b : alignedBoxes)
    {
        vparams->drawTool()->drawBoundingBox({ b[0], b[1] , b[2] }, { b[3], b[4], b[5]}, linesWidth);
    }
    vparams->drawTool()->resetMaterial();

    std::vector<type::Vec3> vertices;
    vertices.reserve(24 * orientedBoxes.size());

    for (const auto& box : orientedBoxes)
    {
        type::vector<type::Vec3> points;
        points.resize(8);
        getPointsFromOrientedBox(box, points);

        vertices.push_back( points[0] );
        vertices.push_back( points[1] );
        vertices.push_back( points[0] );
        vertices.push_back( points[4] );
        vertices.push_back( points[0] );
        vertices.push_back( points[3] );

        vertices.push_back( points[2] );
        vertices.push_back( points[1] );
        vertices.push_back( points[2] );
        vertices.push_back( points[6] );
        vertices.push_back( points[2] );
        vertices.push_back( points[3] );

        vertices.push_back( points[6] );
        vertices.push_back( points[7] );
        vertices.push_back( points[6] );
        vertices.push_back( points[5] );

        vertices.push_back( points[4] );
        vertices.push_back( points[5] );
        vertices.push_back( points[4] );
        vertices.push_back( points[7] );

        vertices.push_back( points[1] );
        vertices.push_back( points[5] );
        vertices.push_back( points[3] );
        vertices.push_back( points[7] );
    }
    vparams->drawTool()->drawLines(vertices, linesWidth, color);
}


template <class DataTypes>
void BoxROI<DataTypes>::roiComputeBBox(const ExecParams* params, type::BoundingBox& bbox)
{
    SOFA_UNUSED(params);

    const vector<type::Vec6>&  alignedBoxes =d_alignedBoxes.getValue();
    const vector<Vec10>& orientedBoxes=d_orientedBoxes.getValue();

    for(const auto& box : alignedBoxes)
    {
        bbox.include({ box[0], box[1], box[2] });
        bbox.include({ box[3], box[4], box[5] });
    }

    for(const auto& box : orientedBoxes)
    {
        vector<type::Vec3> points{};
        points.resize(8);
        getPointsFromOrientedBox(box, points);

        for(int i=0; i<8; i++)
        {
            bbox.include(points[i]);
        }
    }
}


template <class DataTypes>
void BoxROI<DataTypes>::getPointsFromOrientedBox(const Vec10& box, type::vector<type::Vec3>& points) const
{
    points.resize(8);
    points[0] = type::Vec3(box[0], box[1], box[2]);
    points[1] = type::Vec3(box[3], box[4], box[5]);
    points[2] = type::Vec3(box[6], box[7], box[8]);
    const double depth = box[9];

    type::Vec3 normal = (points[1]-points[0]).cross(points[2]-points[0]);
    normal.normalize();

    points[0] += normal * depth/2;
    points[1] += normal * depth/2;
    points[2] += normal * depth/2;

    points[3] = points[0] + (points[2]-points[1]);
    points[4] = points[0] - normal * depth;
    points[6] = points[2] - normal * depth;
    points[5] = points[1] - normal * depth;
    points[7] = points[3] - normal * depth;
}

} // namespace sofa::component::engine::select::boxroi
