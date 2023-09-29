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
#include <sofa/component/engine/generate/config.h>



#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::generate
{

/**
 * This class extrudes a surface
 */
template <class DataTypes>
class ExtrudeSurface : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ExtrudeSurface,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef type::Vec<3,Real> Vec3;

protected:

    ExtrudeSurface();

    ~ExtrudeSurface() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

    bool initialized;
    Data<bool> isVisible; ///< is Visible ?
    Data<Real> heightFactor; ///< Factor for the height of the extrusion (based on normal) ?
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Triangle> > f_triangles; ///< List of triangle indices
    Data<VecCoord> f_extrusionVertices; ///< Position coordinates of the extrusion
    Data<VecCoord> f_surfaceVertices; ///< Position coordinates of the surface
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Triangle> > f_extrusionTriangles; ///< Triangles indices of the extrusion
    Data< type::vector<sofa::core::topology::BaseMeshTopology::TriangleID> > f_surfaceTriangles; ///< Indices of the triangles of the surface to extrude


};

#if !defined(SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API ExtrudeSurface<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::generate
