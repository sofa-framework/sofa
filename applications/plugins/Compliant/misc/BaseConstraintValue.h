#ifndef BASECONSTRAINTVALUE_H
#define BASECONSTRAINTVALUE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "../initCompliant.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   BaseConstraintValue is in charge of producing right-hand side term for
   constraints (the \phi vector in the doc), depending on cases:
   elasticity, hard-stabilized constraints, restitution constraints.

   It exposes values to be mixed by the ODE solver based on
   integration scheme.

*/

class SOFA_Compliant_API BaseConstraintValue : public core::objectmodel::BaseObject
{

  protected:

    typedef core::behavior::BaseMechanicalState mstate_type;
    mstate_type* mstate;

  public:

    SOFA_ABSTRACT_CLASS(BaseConstraintValue, core::objectmodel::BaseObject);


    BaseConstraintValue() : mstate(NULL) {}
    BaseConstraintValue( mstate_type* mstate ) : mstate(mstate) { assert( mstate ); }

    void init()
    {
        if( !mstate )
        {
            mstate = this->getContext()->get<mstate_type>(core::objectmodel::BaseContext::Local);
            assert( mstate );
        }
    }

	// value for stabilization
    virtual void correction(SReal* dst, unsigned n) const = 0;
	
	// value for dynamics
    virtual void dynamics(SReal* dst, unsigned n) const = 0;

};

}
}
}

#endif
