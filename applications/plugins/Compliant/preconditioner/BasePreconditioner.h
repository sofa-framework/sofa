#ifndef COMPLIANT_BASEPRECONDITIONER_H
#define COMPLIANT_BASEPRECONDITIONER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "../initCompliant.h"
#include "../assembly/AssembledSystem.h"

namespace sofa {
namespace component {
namespace linearsolver {

/**
 * 
 *  Base class for linear system preconditioning 
 * 
 * 
 *  Comment coming soon :)
*/

class SOFA_Compliant_API BasePreconditioner : public core::objectmodel::BaseObject
{

  public:

    SOFA_ABSTRACT_CLASS(BasePreconditioner, core::objectmodel::BaseObject);

    virtual void compute( const AssembledSystem::rmat& H ) = 0;

    virtual void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v ) = 0;

};

}
}
}

#endif
 
