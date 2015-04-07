#ifndef COMPLIANT_MISC_PYTHON_H
#define COMPLIANT_MISC_PYTHON_H

#include <sofa/core/objectmodel/Base.h>

namespace impl { 
template<class T> struct argtype;

template<> struct argtype< sofa::core::objectmodel::Base* >  {
    static std::string value() { return "c_void_p"; }
};


template<> struct argtype< int >  {
    static std::string value() { return  "c_int"; }
};

template<> struct argtype< unsigned >  {
    static std::string value() { return  "c_uint"; }
};

template<> struct argtype< float >  {
    static std::string value() { return  "c_float"; }
};

template<> struct argtype< double >  {
    static std::string value() { return  "c_double"; }
};

template<> struct argtype< void >  {
    static std::string value() { return  "None"; }
};

template<> struct argtype< char >  {
    static std::string value() { return  "c_char"; }
};

template<class T> struct argtype< T * > {
    static std::string value() {
        return  "POINTER(" + argtype<T>::value() + ")";
    }
};

template<class T> struct argtype< const T > : argtype<T> { };

// TODO more as needed

template<class Ret>
static std::string make_argtypes( Ret (*func) () ) {
    return "[]";
}

template<class Ret>
static std::string make_restype( Ret (*func) () ) {
    return argtype<Ret>::value();
}


template<class Ret, class Arg>
static std::string make_argtypes( Ret (*func) (Arg) ) {
    return "[" + argtype<Arg>::value() + "]";
}

template<class Ret, class Arg>
static std::string make_restype( Ret (*func) (Arg) ) {
    return argtype<Ret>::value();
}


template<class Ret, class Arg1, class Arg2>
static std::string make_argtypes( Ret (*func) (Arg1, Arg2) ) {
    return "[" + argtype<Arg1>::value() + "," + argtype<Arg2>::value() + "]";
}

template<class Ret, class Arg1, class Arg2>
static std::string make_restype( Ret (*func) (Arg1, Arg2) ) {
    return argtype<Ret>::value();
}

}

// TODO more as needed


class python {

public:

    typedef sofa::core::objectmodel::Base* object;

    typedef void (*func_ptr_type)();
private:
    typedef std::map<func_ptr_type, std::string> map_type;
    static map_type _argtypes, _restype;


    static const char* find(const map_type& map, func_ptr_type ptr) {
        map_type::const_iterator it = map.find( ptr );
        if( it == map.end() ) return 0;
        return it->second.c_str();
    }

public:
    
    template<class Arg>
    static python add(Arg arg) {
        _argtypes[ func_ptr_type(arg) ] = impl::make_argtypes(arg);
        _restype[ func_ptr_type(arg) ] = impl::make_restype(arg);
        return python();
    }


    template<class Arg>
    static const char* argtypes(Arg arg) {
        return find(_argtypes, func_ptr_type(arg) );
    }

    template<class Arg>
    static const char* restype(Arg arg) {
        return find(_restype, func_ptr_type(arg) );
    }

};


#endif
