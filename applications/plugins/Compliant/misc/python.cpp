#include "python.h"

#include <sofa/defaulttype/RigidTypes.h>

python::map_type python::_argtypes;
python::map_type python::_restype;



typedef sofa::core::objectmodel::BaseData* base_data_ptr;
typedef sofa::core::objectmodel::Base* base_ptr;
typedef sofa::core::BaseMapping* base_mapping_ptr;


typedef with_py_callback::py_callback_type py_callback_type;


struct data_pointer {
    void* ptr;
    unsigned size;
};

// extract a pointer to vector data
template<class T>
static data_pointer get_data_thunk(const base_data_ptr& base_data) {

    using namespace sofa::core::objectmodel;
    Data<T>* cast = static_cast< Data<T>* >(base_data);

    const T& value = cast->getValue();
    const void* ptr = value.data();

    data_pointer res;
    res.ptr = const_cast<void*>(ptr);
    res.size = value.size();

    return res;
}

// default case
static data_pointer get_data_thunk_unknown(const base_data_ptr&) {
    std::cerr << "warning: unknown data type" << std::endl;
    
    data_pointer res;
    res.ptr = 0;
    res.size = 0;

    return res;
}

// for_each< F(T1, T2, ...) >::operator() calls f::template
// operator()<T>() for all Ti. F must be default-constructible.
template<class T>
struct for_each;

template<class F>
struct for_each : F {
    
protected:
    // help lookup a little bit
    template<class T>
    void doit() const {

#ifdef _MSC_VER
        // goddamn you visual studio
        this->F::operator()<T>();
#else
        this->F::template operator()<T>();
#endif
        
    }
    
};

template<class F, class T1>
struct for_each<F (T1) > : for_each<F> {
    
    void operator()() const {
        this->template doit<T1>();
    }
};

template<class Tail, class Head>
struct seq : for_each<Tail> {
    
    void operator()() const {
        for_each<Tail>::operator()();
        this->template doit<Head>();
    }
    
};


template<class F, class T1, class T2>
struct for_each<F (T1, T2) > : seq< F(T1), T2 > { };

template<class F, class T1, class T2, class T3>
struct for_each<F (T1, T2, T3) > : seq< F(T1, T2), T3 > { };

template<class F, class T1, class T2, class T3, class T4>
struct for_each<F (T1, T2, T3, T4) > : seq< F(T1, T2, T3), T4 > { };

template<class F, class T1, class T2, class T3, class T4, class T5>
struct for_each<F (T1, T2, T3, T4, T5) > : seq< F(T1, T2, T3, T4), T5 > { };

template<class F, class T1, class T2, class T3, class T4, class T5, class T6>
struct for_each<F (T1, T2, T3, T4, T5, T6) > : seq< F(T1, T2, T3, T4, T5), T6 > { };

template<class F, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct for_each<F (T1, T2, T3, T4, T5, T6, T7) > : seq< F(T1, T2, T3, T4, T5, T6), T7 > { };

// TODO add more as needed


struct vtable {
    
    typedef const std::type_info* key_type;
    typedef data_pointer (*value_type)(const base_data_ptr&);
        
    typedef std::map<key_type, value_type > map_type;

    map_type* map;
    base_data_ptr data;
    
    template<class T>
    void operator()() const {

        using namespace sofa::helper;
        using namespace sofa::core::objectmodel;

        // tests if data is-a Data<vector<T>>, update vtable if so
        if(dynamic_cast< Data<vector<T> >* >(data) ) {
            key_type key = &typeid(*data);
            map->insert( std::make_pair(key, get_data_thunk< vector<T> > ) );
        }
        
    };
    
};


extern "C" {

    const char* argtypes(python::func_ptr_type func) {
        return python::argtypes( func );
    }

    const char* restype(python::func_ptr_type func) {
        return python::restype( func );
    }


    data_pointer get_data_pointer(base_data_ptr base_data) {
        
        static vtable::map_type map;
        
        vtable::key_type key = &typeid(*base_data);
        vtable::map_type::iterator it = map.find(key);

        if(it == map.end()) {
            using namespace sofa::defaulttype;            
            
            for_each< vtable( double,
                              Vec1d, Vec3d, Vec6d,
                              Rigid3dTypes::Deriv, Rigid3dTypes::Coord ) > fill;

            fill.map = &map;
            fill.data = base_data;

            fill();
            
            // defaults to unknown type
            it = map.insert( std::make_pair(key, get_data_thunk_unknown ) ).first;
        }
        
        return it->second(base_data);
    }


    void set_py_callback(base_ptr base, py_callback_type py_callback ) {

        using namespace sofa::defaulttype;        

        with_py_callback* cast = dynamic_cast< with_py_callback* >( base );
        
        if( cast ) {
            cast->py_callback = py_callback;
        } else {
            std::cerr << "error setting python callback" << std::endl;
        }
    }
    
}



with_py_callback::with_py_callback() : py_callback(0) { }
with_py_callback::~with_py_callback() { }


