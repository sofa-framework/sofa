#ifndef COMPLIANT_UTILS_SCOPED_H
#define COMPLIANT_UTILS_SCOPED_H

// some scoped (raii) tools

#include <sofa/helper/AdvancedTimer.h>

namespace scoped
{

template<class T>
class ptr
{
    T* value;

    // copy is hereby prohibited
    ptr(const ptr& ) { }

public:

    ptr( T* value = 0 ) : value( value ) { }

    void reset( T* v = 0 )
    {
        delete value;
        value = v;
    }

    T* operator->() const { return value; }
    T* get() const { return value; }
    T& operator*() const { return *value; }

    ~ptr() { reset(); }

};

struct timer
{
    typedef std::string message_type;
    const message_type message;

    timer( const message_type& message)
        : message(message)
    {
        sofa::helper::AdvancedTimer::stepBegin( message );
    }

    ~timer()
    {
        sofa::helper::AdvancedTimer::stepEnd( message );
    }
};

}

#endif
