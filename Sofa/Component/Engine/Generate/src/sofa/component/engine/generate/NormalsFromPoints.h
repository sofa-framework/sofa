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

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::engine::generate
{

/**
 * This class compute vertex normals by averaging face normals
 */
template <class DataTypes>
class NormalsFromPoints : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NormalsFromPoints,DataTypes),core::DataEngine);
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

protected:

    NormalsFromPoints();

    ~NormalsFromPoints() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data< VecCoord > position; ///< Vertices of the mesh
    Data< type::vector< type::fixed_array <unsigned int,3> > > triangles; ///< Triangles of the mesh
    Data< type::vector< type::fixed_array <unsigned int,4> > > quads; ///< Quads of the mesh
    Data< VecCoord > normals;       ///< result
    Data<bool> invertNormals; ///< Swap normals
    Data<bool> useAngles; ///< Use incident angles to weight faces normal contributions at each vertex
};

#if !defined(SOFA_COMPONENT_ENGINE_NormalsFromPoints_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API NormalsFromPoints<defaulttype::Vec3Types>; 
#endif

} //namespace sofa::component::engine::generate
