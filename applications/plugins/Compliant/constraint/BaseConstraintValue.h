#ifndef BASECONSTRAINTVALUE_H
#define BASECONSTRAINTVALUE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <Compliant/config.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa {
namespace component {
namespace odesolver {

/**

   BaseConstraintValue is in charge of producing right-hand side term for
   constraints (the \phi vector in the doc).
    Function dynamics(SReal* dst, unsigned n) produces the right-hand term for ODE solution, while function correction(SReal* dst, unsigned n) produces the right-hand term for velocity correction.


    These will have different implementations depending on cases: elasticity, hard-stabilized constraints, restitution constraints.


    @author Maxime Tournier and Matthieu Nesme, 2013

*/

class SOFA_Compliant_API BaseConstraintValue : public core::objectmodel::BaseObject
{

  protected:

    typedef core::behavior::BaseMechanicalState mstate_type;
	mstate_type::SPtr mstate;
	
  public:

    SOFA_ABSTRACT_CLASS(BaseConstraintValue, core::objectmodel::BaseObject);


    BaseConstraintValue( mstate_type* mstate = 0 ) 
		: mstate(mstate) { 
		
	}

    void init() override
    {
        if( !mstate )
        {
            mstate = this->getContext()->get<mstate_type>(core::objectmodel::BaseContext::Local);
            assert( mstate );
        }

    }

    /// Value for stabilization: right-hand term for velocity correction
    /// @param n nb constraint blocks
    /// @param dim nb lines per constraint
    virtual void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const = 0;
	
    /// Value for dynamics: right-hand term for time integration.
    /// @param n nb constraint blocks
    /// @param dim nb lines per constraint
    /// @param stabilization tells if the solver is performing the correction pass (ie if the correction value is used). Otherwise the constraint must be fully corrected by the dynamics (not stabilized constraint)
    virtual void dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const = 0;

    /// Testing which constraint is violated based on the violation @param posId
    /// Should NOT be called for bilateral constraints that can always be considered as violated.
    /// @param activateMask is a per constraint block mask, flagging active constraint
    /// In general even non-violated constraints remain active,
    /// but in some case (eg Restitution), they must be deactivated
    /// @param n nb constraint blocks
    /// @param dim nb lines per constraint
    virtual void filterConstraints( helper::vector<bool>*& /*activateMask*/, const core::MultiVecCoordId& /*posId*/, unsigned /*n*/, unsigned /*dim*/ ) { /*all constraints are active by default*/ }

    /// clear an eventual violated mask
    virtual void clear() {}


};

}
}
}

#endif
