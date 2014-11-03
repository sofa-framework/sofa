#ifndef COMPLIANT_COULOMBCONSTRAINT_H
#define COMPLIANT_COULOMBCONSTRAINT_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {


/// A Coulomb Cone Friction constraint
struct SOFA_Compliant_API CoulombConstraint : Constraint {
	
    SOFA_CLASS(CoulombConstraint, Constraint);

	// friction coefficient f_T <= mu. f_N
	SReal mu;

    CoulombConstraint( SReal mu = 1.0 );

    virtual void project( SReal* out, unsigned n, unsigned index, bool correctionPass=false ) const;


    bool horizontalProjection; ///< should the projection be horizontal? By default a regular orthogonal cone projection is performed.
	
};

}
}
}


#endif // COMPLIANT_COULOMBCONSTRAINT_H
