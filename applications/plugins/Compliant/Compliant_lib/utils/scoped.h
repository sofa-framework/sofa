#ifndef COMPLIANT_UTILS_SCOPED_H
#define COMPLIANT_UTILS_SCOPED_H

// a simple scoped pointer.

template<class T>
class scoped
{
    T* value;

    // copy is hereby prohibited
    scoped(const scoped& ) { }

public:

    scoped( T* value = 0 ) : value( value ) { }

    void reset( T* v = 0 )
    {
        delete value;
        value = v;
    }

    T* operator->() const { return value; }
    T* get() const { return value; }
    T& operator*() const { return *value; }

    ~scoped() { reset(); }

};


#endif
