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
#ifndef SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_H
#define SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class returns the indices given a list of values.
 */
template <class T>
class IndicesFromValues : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IndicesFromValues,T),core::DataEngine);
    typedef T Value;
    typedef sofa::helper::vector<T> VecValue;
    typedef unsigned int Index;
    typedef sofa::helper::vector<Index> VecIndex;

protected:

    IndicesFromValues();

    virtual ~IndicesFromValues();
public:
    void init() override;

    void reinit() override;

    void update() override;

    core::objectmodel::Data<VecValue> f_values;
    core::objectmodel::Data<VecValue> f_global;
    core::objectmodel::Data<VecIndex> f_indices;
    core::objectmodel::Data<VecIndex> f_otherIndices;
    core::objectmodel::Data<bool> f_recursiveSearch;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_CPP)
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<std::string>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<int>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<unsigned int>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 2> >;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 3> >;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 4> >;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 8> >;
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<double>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec2d>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec3d>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Coord>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Deriv>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Coord>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Deriv>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<float>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec2f>;
extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec3f>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Coord>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Deriv>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Coord>;
// extern template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Deriv>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
