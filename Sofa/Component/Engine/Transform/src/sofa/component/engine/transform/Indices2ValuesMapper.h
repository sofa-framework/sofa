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
#include <sofa/component/engine/transform/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/type/vector.h>


namespace sofa::component::engine::transform
{


template <class DataTypes>
class Indices2ValuesMapper : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Indices2ValuesMapper,DataTypes),sofa::core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::type::Vec<3,Real> Vec3;
    typedef unsigned int Index;

protected:

    Indices2ValuesMapper();
    ~Indices2ValuesMapper() override {}
public:
    void init() override;
    void reinit() override;
    void doUpdate() override;

    //Input
    Data<sofa::type::vector<Real> > f_inputValues; ///< Already existing values (can be empty) 
    Data<sofa::type::vector<Real> > f_indices; ///< Indices to map value on 
    Data<sofa::type::vector<Real> > f_values; ///< Values to map indices on 

    //Output
    Data<sofa::type::vector<Real> > f_outputValues; ///< New map between indices and values

    //Parameter
    Data<Real> p_defaultValue; ///< Default value for indices without any value

};

#if !defined(SOFA_COMPONENT_ENGINE_INDICES2VALUESMAPPER_CPP)
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API Indices2ValuesMapper<sofa::defaulttype::Vec3Types>;
 
#endif


} //namespace sofa::component::engine::transform
