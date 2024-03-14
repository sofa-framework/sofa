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

#include <set>

namespace sofa::component::engine::generate
{

/**
 * This class merge 2 coordinate vectors.
 */
template <class T>
class MergeSets : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MergeSets,T),core::DataEngine);
    typedef T Index;
    typedef sofa::type::vector<T> VecIndex;
    typedef std::set<T> SetIndex;

protected:
    MergeSets();

    ~MergeSets() override;
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<VecIndex> f_in1; ///< first set of indices
    Data<VecIndex> f_in2; ///< second set of indices
    Data<VecIndex> f_out; ///< merged set of indices
    Data<std::string> f_op; ///< name of operation to compute (union, intersection, difference, symmetric_difference)
};

#if !defined(SOFA_COMPONENT_ENGINE_MERGESETS_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeSets<int>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeSets<unsigned int>;
//extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeSets<long long>;
//extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeSets<unsigned long long>;
#endif

} //namespace sofa::component::engine::generate
