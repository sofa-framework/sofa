#include "UserCoulombConstraint.h"
#include <sofa/core/ObjectFactory.h>
#include <Compliant/utils/cone.h>

namespace sofa {
namespace component {
namespace linearsolver {

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(UserCoulombConstraint);

static int UserCoulombConstraintClass = 
    core::RegisterObject("user-friendly coulomb constraint")
    .add< UserCoulombConstraint >();

UserCoulombConstraint::UserCoulombConstraint()
    : mu(initData(&mu, 1.0, "mu", "friction coefficient")),
      normal(initData(&normal, {0, 1, 0}, "normal", "normal vector in world frame")),
      horizontal(initData(&horizontal, true, "horizontal", "horizontal/orthogonal projection")) {
    
}


void UserCoulombConstraint::project( SReal* out, unsigned n, unsigned /*index*/, bool correct) const {
    // correction does not project (used for inverse dynamics correction)
    if(correct) return;

    // standard horizontal cone projection for dynamics
    enum {N = 3};

    assert( n % N  == 0);

    typedef Eigen::Matrix<SReal, 3, 1> vec3;
    typedef Eigen::Map< vec3 > view_type;
    typedef Eigen::Map< const vec3 > const_view_type;        

    // TODO multiple normals?
    const vec3 normal = const_view_type(this->normal.getValue().data());

    const SReal mu = this->mu.getValue();
    
    for(SReal* it = out, *last = out + n; it < last; it += N) {
        view_type view(it);
        if(horizontal.getValue()) {
            view = cone_horizontal<SReal>(view, normal, mu);
        } else {
            view = cone<SReal>(view, normal, mu);
        }
    }
    
}


std::size_t UserCoulombConstraint::getConstraintTypeIndex() const {
    return constraint_index(this);
}



}
}
}

