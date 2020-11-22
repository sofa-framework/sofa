#ifndef COMPLIANT_ResistanceConstraint_H
#define COMPLIANT_ResistanceConstraint_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {


/// A constraint creating resistance in a joint
/// a cinematic dof is free to move only if the force creating the movement is > threshold
/// the maximum amount of constraint force is threshold
/// Must be used with a ConstraintValue that constrain the velocity to 0 and that is not stabilizable (e.g. VelocityConstraintValue with velocities="0")
///
/// @author Matthieu Nesme
///
struct SOFA_Compliant_API ResistanceConstraint : Constraint {
	
    SOFA_CLASS(ResistanceConstraint, Constraint);
    SOFA_COMPLIANT_CONSTRAINT_H( ResistanceConstraint )


    ResistanceConstraint();

    void project( SReal* out, unsigned n, unsigned index, bool correctionPass=false ) const override;


    Data<SReal> d_threshold; ///< the resistance force

};

}
}
}


#endif // COMPLIANT_ResistanceConstraint_H
