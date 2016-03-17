%module defaulttype

%include "std_string.i"

%{
#include "sofa/defaulttype/Vec.h"
%}

namespace sofa
{
namespace defaulttype
{

template <int N, typename real=float>
class Vec
{
public:
typedef std::size_t size_type;
    
Vec();
//real& operator[](size_type i);

};

// TODO take care of double/float
//%template(Vector1) Vec<1,double>;
//%template(Vector2) Vec<2,double>;
%template(Vector3) Vec<3,double>;
//%template(Vector4) Vec<4,double>;

// TODO: is there a way not to extend each specialization at once ?
%extend Vec<3,double> {
    double __getitem__(size_type i) {
        return (*self)[i];
    }
    void __setitem__(size_type i, double value) {
        (*self)[i]=value;
    }
    std::string __str__() {
        std::ostringstream ss;
        ss << *self;
        return ss.str();
    }
}

}
}

