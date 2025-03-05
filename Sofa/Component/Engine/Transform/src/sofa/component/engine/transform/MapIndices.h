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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::engine::transform
{

/**
 * This class apply a permutation to a set of indices
 */
template <class T>
class MapIndices : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MapIndices,T),core::DataEngine);
    typedef T Value;
    typedef sofa::type::vector<T> VecValue;
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);
    typedef sofa::type::vector<sofa::Index> VecIndex;
    typedef std::map<sofa::Index, sofa::Index> MapIndex;
protected:
    MapIndices();

    ~MapIndices() override;
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    core::objectmodel::Data<VecValue> f_in; ///< input indices
    core::objectmodel::Data<VecIndex> f_indices; ///< array containing in ith cell the input index corresponding to the output index i (or reversively if transpose=true)
    core::objectmodel::Data<VecValue> f_out; ///< Output indices
    core::objectmodel::Data<std::string> f_outStr; ///< Output indices, converted as a string
    core::objectmodel::Data<bool> f_transpose; ///< Should the transposed mapping be used ?

    template<class V>
    void applyIndex(V& v, const MapIndex& m)
    {
        const typename MapIndex::const_iterator it = m.find(v);
        if (it != m.end())
            v = it->second;
    }

    void apply(Value& v, const MapIndex& m);
};

#if !defined(SOFA_COMPONENT_ENGINE_MAPINDICES_CPP)
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices<int>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices<unsigned int>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 2> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 3> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 4> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 8> >;
#endif

} //namespace sofa::component::engine::transform
