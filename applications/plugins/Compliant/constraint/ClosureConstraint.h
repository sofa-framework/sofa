#ifndef COMPLIANT_CLOSURECONSTRAINT_H
#define COMPLIANT_CLOSURECONSTRAINT_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {


/// Does nothing! I.e. is is a bilateral constraint.
/// But it allows to differenciate regular bilateral constraints
/// from manually set closure constraints.
/// The sub-system (dynamics+bilaterals) used in certain solvers
/// can then be solved in linear-time.
/// cf Baraff 96 Linear-Time Dynamics using Lagrange Multipliers
///
/// @author Matthieu Nesme
struct SOFA_Compliant_API ClosureConstraint : Constraint {
	
    SOFA_CLASS(ClosureConstraint, Constraint);
    SOFA_COMPLIANT_CONSTRAINT_H( ClosureConstraint )
	
    void project(SReal*, unsigned, unsigned, bool=false) const override {}
	
};


}
}
}

#endif
