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
 * This class extrudes a quad surface into a set of hexahedra
 */
template <class DataTypes>
class ExtrudeQuadsAndGenerateHexas : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ExtrudeQuadsAndGenerateHexas,DataTypes),core::DataEngine);

    typedef typename DataTypes::Coord     Coord;
    typedef typename DataTypes::VecCoord  VecCoord;
    typedef typename DataTypes::Real      Real;
    typedef type::Vec<3,Real>      Vec3;

protected:

    ExtrudeQuadsAndGenerateHexas();

    ~ExtrudeQuadsAndGenerateHexas() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw( const core::visual::VisualParams* ) override;

    bool                                             initialized;
    Data<bool>                                       isVisible; ///< is Visible ?
    Data<Coord>                                      f_scale; ///< Apply a scaling factor to the extruded mesh
    Data<Real>                                       f_thickness;
    Data<Real>                                       f_thicknessIn; ///< Thickness of the extruded volume in the opposite direction of the normals
    Data<Real>                                       f_thicknessOut; ///< Thickness of the extruded volume in the direction of the normals
    Data<int>                                        f_numberOfSlices; ///< Number of slices / steps in the extrusion
    Data<bool>                                       f_flipNormals; ///< If true, will inverse point order when creating hexa
    Data<VecCoord>                                   f_surfaceVertices; ///< Position coordinates of the surface
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Quad> >   f_surfaceQuads; ///< Indices of the quads of the surface to extrude
    Data<VecCoord>                                   f_extrudedVertices; ///< Coordinates of the extruded vertices
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Quad> >   f_extrudedSurfaceQuads; ///< List of new surface quads generated during the extrusion
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Quad> >   f_extrudedQuads; ///< List of all quads generated during the extrusion
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Hexa> >   f_extrudedHexas; ///< List of hexahedra generated during the extrusion
};

#if !defined(SOFA_COMPONENT_ENGINE_EXTRUDEQUADSANDGENERATEHEXAS_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API ExtrudeQuadsAndGenerateHexas<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::generate
