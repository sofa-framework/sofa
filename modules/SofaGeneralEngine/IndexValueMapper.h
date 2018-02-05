/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef INDEXVALUEMAPPER_H_
#define INDEXVALUEMAPPER_H_
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
class IndexValueMapper : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IndexValueMapper,DataTypes),sofa::core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef unsigned int Index;

protected:

    IndexValueMapper();
    ~IndexValueMapper() {}
public:
    virtual void init() override;
    virtual void reinit() override;
    virtual void update() override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const IndexValueMapper<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Input
    Data<sofa::helper::vector<Real> > f_inputValues; ///< Already existing values (can be empty)
    Data<sofa::helper::vector<Index> > f_indices; ///< Indices to map value on
    Data<Real> f_value; ///< Value to map indices on

    //Output
    Data<sofa::helper::vector<Real> > f_outputValues; ///< New map between indices and values

    //Parameter
    Data<Real> p_defaultValue; ///< Default value for indices without any value

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(INDEXVALUEMAPPER_CPP_)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API IndexValueMapper<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API IndexValueMapper<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif /* INDEXVALUEMAPPER_H_ */
