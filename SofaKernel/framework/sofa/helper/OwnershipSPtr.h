/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef __SOFA_HELPER_OWNERSHIPSPTR_H__
#define __SOFA_HELPER_OWNERSHIPSPTR_H__

#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa {

namespace helper {


/// Smart pointer where the user precises if it must take the ownership (and so
/// be in charge of deleting the data).
/// Either it can point to an existing data without taking the ownership
/// or it can point to a new temporary Data that will be deleted when this
/// smart pointer is deleted (taking ownership).
/// @warning Maybe an equivalent smart pointer exists in stl or boost that I do not know
/// @author Matthieu Nesme
template<class T>
class OwnershipSPtr
{

    const T* t; ///< the pointed data (const)
    mutable bool ownership; ///< does this smart pointer have the ownership (and must delete the pointed data)?

public:

    /// default constructor: no pointed data, no ownership
    OwnershipSPtr() : t(NULL), ownership(false) {}

    /// point to a data, manually set ownership
    OwnershipSPtr( const T* t, bool ownership ) : t(t), ownership(ownership) {}

    /// copy constructor that steals the ownership if 'other' had it
    OwnershipSPtr( const OwnershipSPtr<T>& other ) : t(other.t), ownership(other.ownership) { other.ownership=false; }

    /// destructor will delete the data only if it has the ownership
    ~OwnershipSPtr() { if( ownership ) delete t; }

    /// copy operator is stealing the ownership if 'other' had it
    void operator=(const OwnershipSPtr<T>& other) { t=other.t; ownership=other.ownership; other.ownership=false; }

    /// get a const ref to the pointed data
    const T& operator*() const { return *t; }

    /// get a const pointer to the pointer data
    const T* operator->() const { return t; }

};



} // namespace helper


} // namespace sofa

#endif
