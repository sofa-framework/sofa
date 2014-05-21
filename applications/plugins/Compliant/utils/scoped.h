#ifndef COMPLIANT_UTILS_SCOPED_H
#define COMPLIANT_UTILS_SCOPED_H

// some scoped (RAII) tools

// author: maxime.tournier@inria.fr
// license: LGPL 2.1

#include <sofa/helper/AdvancedTimer.h>

namespace scoped {

    // scoped pointer similar to std::unique_ptr without move semantics
    template<class T>
    class ptr {
        T* value;

        // copy is hereby prohibited
        ptr(const ptr& ) { }

    public:

        ptr( T* value = 0 ) : value( value ) { }

        void reset( T* v = 0 ) {
            delete value;
            value = v;
        }

        T* operator->() const { return value; }
        T* get() const { return value; }
        T& operator*() const { return *value; }

        ~ptr() { reset(); }
    };




    /// @ deprecated, use sofa::helper::ScopedAdvancedTimer instead
    ////// the code has been MOVED to AdvancedTimer.h
    typedef sofa::helper::ScopedAdvancedTimer timer;

}



#endif
