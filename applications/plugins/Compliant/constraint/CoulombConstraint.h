#ifndef COMPLIANT_COULOMBCONSTRAINT_H
#define COMPLIANT_COULOMBCONSTRAINT_H

#include "Constraint.h"
#include <sofa/helper/template_name.h>

namespace sofa {
namespace component {
namespace linearsolver {


struct SOFA_Compliant_API CoulombConstraintBase : Constraint {
    SOFA_COMPLIANT_CONSTRAINT_H( CoulombConstraintBase );

    // coulomb friction coefficient f_T <= mu. f_N    
    virtual SReal frictionCoefficient() const = 0;
    
};

/// A standard Coulomb Cone Friction constraint (normal along x)
template<class DataTypes>
struct SOFA_Compliant_API CoulombConstraint : CoulombConstraintBase {

    SReal mu;

    SReal frictionCoefficient() const;
    
    SOFA_CLASS(SOFA_TEMPLATE(CoulombConstraint, DataTypes), Constraint);

    CoulombConstraint( SReal mu = 1.0 );

    // WARNING index is not used (see Constraint.h)
    virtual void project( SReal* out, unsigned n, unsigned /*index*/,
                          bool correctionPass=false ) const;


    ///< should the projection be horizontal (default)? Otherwise an orthogonal
    ///cone projection is performed.
    bool horizontalProjection; 

    static std::string templateName(const CoulombConstraint* self) {
        const static std::string name = helper::template_name(self);
        return name;
    }

    std::string getTemplateName() const { return templateName(this); }
    
};




}
}
}


#endif // COMPLIANT_COULOMBCONSTRAINT_H
