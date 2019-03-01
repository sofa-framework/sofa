#ifndef COMPLIANT_COULOMBCONSTRAINT_H
#define COMPLIANT_COULOMBCONSTRAINT_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {


struct SOFA_Compliant_API CoulombConstraintBase : Constraint
{
    SOFA_COMPLIANT_CONSTRAINT_H( CoulombConstraintBase )

    // friction coefficient f_T <= mu. f_N
    SReal mu;
};

/// A Coulomb Cone Friction constraint
template<class DataTypes>
struct SOFA_Compliant_API CoulombConstraint : CoulombConstraintBase {
	
    SOFA_CLASS(SOFA_TEMPLATE(CoulombConstraint, DataTypes), Constraint);

    CoulombConstraint( SReal mu = 1.0 );

    // WARNING index is not used (see Constraint.h)
    void project( SReal* out, unsigned n, unsigned /*index*/,
                          bool correctionPass=false ) const override;


    bool horizontalProjection; ///< should the projection be horizontal (default)? Otherwise an orthogonal cone projection is performed.
	
};

}
}
}


#endif // COMPLIANT_COULOMBCONSTRAINT_H
