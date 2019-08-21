#ifndef COMPLIANT_CONSTRAINT_H
#define COMPLIANT_CONSTRAINT_H

#include <Compliant/config.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {


/// This has to be added in the Constraint class definition (as public).
/// It is a way to check constraint type for cheap.
#define SOFA_COMPLIANT_CONSTRAINT_H(T) \
    protected:\
    static const size_t s_constraintTypeIndex; \
    public:\
    virtual size_t getConstraintTypeIndex() const { return T::s_constraintTypeIndex; } \
    static bool checkConstraintType( const Constraint* constraint ) { return constraint->getConstraintTypeIndex() == T::s_constraintTypeIndex; }


/// This has to be added in the Constraint implementation file.
/// It is a way to check constraint type for cheap.
#define SOFA_COMPLIANT_CONSTRAINT_CPP(T) \
    const size_t T::s_constraintTypeIndex = ++sofa::component::linearsolver::Constraint::s_lastConstraintTypeIndex;


/// Base class to define the constraint type
struct SOFA_Compliant_API Constraint : public core::objectmodel::BaseObject {
    
    SOFA_ABSTRACT_CLASS(Constraint, sofa::core::objectmodel::BaseObject);

    Constraint();

    /// project the response on the valid sub-space
    /// @param out: the buffer to project
    /// @param n: the total size of the buffer
    /// @param index: unused in the general case (introduced for the 6d contact work)
    /// @param correctionPass informs if the correction pass is performing (in which case only a friction projection should only treat the unilateral projection for example)
    virtual void project(SReal* out, unsigned n, unsigned index, bool correctionPass=false) const = 0;

    /// Flagging which constraints must be activated (true == active)
    /// ie filter out all deactivated constraints (force lambda to 0)
    /// If mask is NULL or empty, all constraints are activated
    /// A value per constraint block (NOT per constraint line)
    helper::vector<bool>* mask;
	

    /// \returns unique type index
    /// for fast Constraint type comparison with unique indices (see function 'checkConstraintType')
    /// @warning this mechanism will only work for the last derivated type (and not for eventual intermediaries)
    /// e.g. for C derivated from B derivated from A, checkConstraintType will returns true only for C* but false for B* or A*
    /// Should be implemented by using macros SOFA_COMPLIANT_CONSTRAINT_H / SOFA_COMPLIANT_CONSTRAINT_CPP
    virtual size_t getConstraintTypeIndex() const = 0;

protected:
    static size_t s_lastConstraintTypeIndex; ///< storing the last given id
};

}
}
}


#endif // COMPLIANT_CONSTRAINT_H
