/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_H
#define SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/component.h>

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

public:

    IndicesFromValues();

    virtual ~IndicesFromValues();

    void init();

    void reinit();

    void update();

    core::objectmodel::Data<VecValue> f_values;
    core::objectmodel::Data<VecValue> f_global;
    core::objectmodel::Data<VecIndex> f_indices;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_CPP)
#pragma warning(disable : 4231)
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<std::string>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<int>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<unsigned int>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 2> >;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 3> >;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 4> >;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 8> >;
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<double>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Vec2d>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Vec3d>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Coord>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Deriv>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Coord>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Deriv>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<float>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Vec2f>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Vec3f>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Coord>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Deriv>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Coord>;
template class SOFA_COMPONENT_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Deriv>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
