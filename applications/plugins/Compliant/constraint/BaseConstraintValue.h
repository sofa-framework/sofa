#ifndef BASECONSTRAINTVALUE_H
#define BASECONSTRAINTVALUE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "../initCompliant.h"

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

    /// Value for stabilization: right-hand term for velocity correction
    virtual void correction(SReal* dst, unsigned n) const = 0;
	
    /// Value for dynamics: right-hand term for time integration.
    virtual void dynamics(SReal* dst, unsigned n) const = 0;

};

}
}
}

#endif
