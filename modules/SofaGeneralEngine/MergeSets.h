/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MERGESETS_H
#define SOFA_COMPONENT_ENGINE_MERGESETS_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <set>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef sofa::helper::vector<T> VecIndex;
    typedef std::set<T> SetIndex;

protected:
    MergeSets();

    virtual ~MergeSets();
public:
    void init();

    void reinit();

    void update();

    Data<VecIndex> f_in1;
    Data<VecIndex> f_in2;
    Data<VecIndex> f_out;
    Data<std::string> f_op;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MERGESETS_CPP)
extern template class SOFA_ENGINE_API MergeSets<int>;
extern template class SOFA_ENGINE_API MergeSets<unsigned int>;
//extern template class SOFA_ENGINE_API MergeSets<long long>;
//extern template class SOFA_ENGINE_API MergeSets<unsigned long long>;
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
