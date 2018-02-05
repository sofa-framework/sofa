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
#ifndef SOFA_CORE_OBJECTMODEL_SPTR_H
#define SOFA_CORE_OBJECTMODEL_SPTR_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/vector.h>
#include <sofa/core/core.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief new operator for classes with smart pointers (such as all components deriving from Base)
 *
 *  This class should be used as :
 *     MyT::SPtr p = sofa::core::objectmodel::New<MyT>(myargs);
 *  instead of :
 *     MyT* p = new MyT(myargs);
 *
 *  The use of this New operator and SPtr pointers insures that all created objects are :
 *    - destroyed (no leak),
 *    - only once (no double desctructions),
 *    - and only after the last reference to them are erased (no invalid pointers).
 *
 */
template<class T>
class New : public T::SPtr {
    typedef typename T::SPtr SPtr;
public:
    template<class ... Args>
    New(Args&& ... args) : SPtr( new T(std::forward<Args>(args)...) ) { }
};

/// dynamic_cast operator for SPtr
template<class T>
class SPtr_dynamic_cast : public T::SPtr
{
public:
    template<class UPtr>
    SPtr_dynamic_cast(UPtr p) : T::SPtr(dynamic_cast<T*>(p.get())) {}
};

/// static_cast operator for SPtr
template<class T>
class SPtr_static_cast : public T::SPtr
{
public:
    template<class UPtr>
    SPtr_static_cast(UPtr p) : T::SPtr(static_cast<T*>(p.get())) {}
};

/// const_cast operator for SPtr
template<class T>
class SPtr_const_cast : public T::SPtr
{
public:
    template<class UPtr>
    SPtr_const_cast(UPtr p) : T::SPtr(const_cast<T*>(p.get())) {}
};

} // namespace objectmodel

} // namespace core

} // namespace sofa



#endif

