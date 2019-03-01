/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_H
#define SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{



/**
 * This class returns the values given a list of indices.
 */
template <class T>
class ValuesFromIndices : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ValuesFromIndices,T),core::DataEngine);
    typedef T Value;
    typedef sofa::helper::vector<T> VecValue;
    typedef unsigned int Index;
    typedef sofa::helper::vector<Index> VecIndex;

protected:

    ValuesFromIndices();

    virtual ~ValuesFromIndices();
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<VecValue> f_in; ///< input values
    Data<VecIndex> f_indices; ///< Indices of the values
    Data<VecValue> f_out; ///< Output values corresponding to the indices
    Data<std::string> f_outStr; ///< Output values corresponding to the indices, converted as a string

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const ValuesFromIndices<T>* = NULL)
    {
        return sofa::defaulttype::DataTypeName<T>::name();
    }
};

#if  !defined(SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_CPP)
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<std::string>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<int>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<unsigned int>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices< helper::fixed_array<unsigned int, 2> >;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices< helper::fixed_array<unsigned int, 3> >;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices< helper::fixed_array<unsigned int, 4> >;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices< helper::fixed_array<unsigned int, 8> >;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<double>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Vec2d>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Vec3d>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Vec4d>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Vec6d>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Rigid2Types::Coord>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Rigid2Types::Deriv>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Rigid3Types::Coord>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromIndices<defaulttype::Rigid3Types::Deriv>;
 
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
