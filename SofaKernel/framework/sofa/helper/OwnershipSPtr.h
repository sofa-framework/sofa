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
