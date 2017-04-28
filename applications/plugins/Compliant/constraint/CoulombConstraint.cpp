#include "CoulombConstraint.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

using namespace sofa::defaulttype;

SOFA_COMPLIANT_CONSTRAINT_CPP(CoulombConstraintBase)


static int CoulombConstraintClass = 
    core::RegisterObject("standard coulomb constraint")
    
#ifndef SOFA_FLOAT
    .add< CoulombConstraint< Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< CoulombConstraint< Vec3fTypes > >()
#endif
    ;

// TODO register in the factory in case we want to use it manually in
// the graph

#ifndef SOFA_FLOAT
template struct SOFA_Compliant_API CoulombConstraint<Vec3dTypes>;
template struct SOFA_Compliant_API CoulombConstraint<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_Compliant_API CoulombConstraint<Vec3fTypes>;
template struct SOFA_Compliant_API CoulombConstraint<Vec6fTypes>;
#endif


SOFA_DECL_CLASS(UserCoulombConstraint);

static int UserCoulombConstraintClass = 
    core::RegisterObject("arbitrary coulomb constraint")
    .add<UserCoulombConstraint>();


UserCoulombConstraint::UserCoulombConstraint()
    : mu(initData(&mu, 1.0, "mu", "friction coefficient")),
      normal(initData(&normal, {0, 1, 0}, "normal", "normal vector in world frame")) {
    
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

    // TODO multiple normals
    const vec3 normal = const_view_type(this->normal.getValue().data());

    const SReal mu = this->mu.getValue();
    
    for(SReal* it = out, *last = out + n; it < last; it += N) {
        view_type view(it);
        view = cone_horizontal<SReal>(view, normal, mu);
    }
    
}


std::size_t UserCoulombConstraint::getConstraintTypeIndex() const {
    struct stealer : CoulombConstraintBase {
        using CoulombConstraintBase::s_constraintTypeIndex;
    };
    return stealer::s_constraintTypeIndex;
    return constraint_index(this);
}



}
}
}

