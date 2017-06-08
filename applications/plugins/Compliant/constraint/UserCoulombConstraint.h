#ifndef COMPLIANT_USERCOULOMBCONSTRAINT_H
#define COMPLIANT_USERCOULOMBCONSTRAINT_H

#include <Compliant/constraint/CoulombConstraint.h>

#include <sofa/helper/template_name.h>

namespace sofa {
namespace component {
namespace linearsolver {

// a more general friction constraint: user-defined normal direction, mu in a
// data, projection method, etc.
struct UserCoulombConstraint : CoulombConstraintBase {
    SOFA_CLASS(UserCoulombConstraint, Constraint);

    SReal frictionCoefficient() const;
    
    Data<SReal> mu;

    using normal_type = defaulttype::Vec<3, SReal>;
    Data<normal_type> normal;

    Data<bool> horizontal;
    
    UserCoulombConstraint();

    // WARNING index is not used (see Constraint.h)
    virtual void project( SReal* out, unsigned n, unsigned /*index*/, bool correct) const;
    
    virtual std::size_t getConstraintTypeIndex() const;
};


}
}
}


#endif // COMPLIANT_COULOMBCONSTRAINT_H
