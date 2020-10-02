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
#ifndef SOFA_COMPONENT_ENGINE_GENERATEGRID_H
#define SOFA_COMPONENT_ENGINE_GENERATEGRID_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class generates a cylinder mesh given its radius, length and height. The output  mesh is composed of tetrahedra elements
 */
template <class DataTypes>
class GenerateGrid : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GenerateGrid,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
	typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;
    typedef typename SeqTetrahedra::value_type Tetrahedron;
    typedef typename SeqHexahedra::value_type Hexahedron;

public:

    GenerateGrid();

    ~GenerateGrid() {}

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const GenerateGrid<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    Data<VecCoord> f_outputX; ///< ouput position
    Data<SeqTetrahedra> f_tetrahedron; ///< output tetrahedra
    Data<SeqHexahedra> f_hexahedron; ///< output hexahedron
    Data<Real > f_length; /// length of each cube 
    Data<Real > f_height; /// height of each cube
    Data<Real > f_width; /// width of each cube
    Data<Coord> f_origin; /// origin
    Data<size_t> f_resolutionLength; /// number of cubes in the length direction
    Data<size_t> f_resolutionWidth; /// number of cubes in the width direction
    Data<size_t> f_resolutionHeight; /// number of cubes in the height direction
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_GENERATEGRID_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API GenerateGrid<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API GenerateGrid<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
