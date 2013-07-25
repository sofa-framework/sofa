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


    // use this to "log time" of a given scope
    struct timer {
        typedef std::string message_type;
        const message_type message;

        timer( const message_type& message)
        : message(message) {
            sofa::helper::AdvancedTimer::stepBegin( message );
        }

        ~timer() {
            sofa::helper::AdvancedTimer::stepEnd( message );
        }
    };

}



#endif
