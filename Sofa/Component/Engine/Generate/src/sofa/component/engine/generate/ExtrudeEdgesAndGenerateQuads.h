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
 * This engine extrudes an edge-based curve into a quad surface patch
 */
template <class DataTypes>
class ExtrudeEdgesAndGenerateQuads : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ExtrudeEdgesAndGenerateQuads,DataTypes),core::DataEngine);

    typedef typename DataTypes::Coord     Coord;
    typedef typename DataTypes::VecCoord  VecCoord;
    typedef typename DataTypes::Real      Real;
    typedef type::Vec<3,Real>      Vec3;

protected:

    ExtrudeEdgesAndGenerateQuads();

    ~ExtrudeEdgesAndGenerateQuads() override {}
public:

    void init() override;
    void reinit() override;
    void doUpdate() override;

    bool                                             initialized;
    Data<Coord>                                      d_direction; ///< Direction along which to extrude the curve
    Data<Real>                                       d_thickness;
    Data<Real>                                       d_thicknessIn; ///< Thickness of the extruded volume in the opposite direction of the normals
    Data<Real>                                       d_thicknessOut; ///< Thickness of the extruded volume in the direction of the normals
    Data<int>                                        d_nbSections; ///< Number of sections / steps in the extrusion
    Data<VecCoord>                                   d_curveVertices; ///< Position coordinates along the initial curve
    Data<type::vector<sofa::core::topology::BaseMeshTopology::Edge> >   d_curveEdges; ///< Indices of the edges of the curve to extrude
    Data<VecCoord>                                   d_extrudedVertices; ///< Coordinates of the extruded vertices
    Data<type::vector<sofa::core::topology::BaseMeshTopology::Edge> >   d_extrudedEdges; ///< List of all edges generated during the extrusion
    Data<type::vector<sofa::core::topology::BaseMeshTopology::Quad> >   d_extrudedQuads; ///< List of all quads generated during the extrusion

protected:

    void checkInput();
};

#if !defined(SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API ExtrudeEdgesAndGenerateQuads<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::generate
