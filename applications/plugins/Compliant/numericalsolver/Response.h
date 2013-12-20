#ifndef COMPLIANT_RESPONSE_H
#define COMPLIANT_RESPONSE_H

#include "initCompliant.h"

#include "assembly/AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {

struct Response : core::objectmodel::BaseObject {

	SOFA_CLASS(Response, core::objectmodel::BaseObject);

	typedef AssembledSystem system_type;
	
	typedef system_type::real real;
	typedef system_type::vec vec;

	typedef system_type::mat mat;
	typedef system_type::cmat cmat;


    Response()
        : core::objectmodel::BaseObject()
        , _constant( initData( &_constant, false, "constant", "If true, the factorization will be computed only once") )
        , _firstFactorization( true )
    {}

	
	virtual void factor(const mat& ) = 0;
	
	virtual void solve(cmat&, const cmat& ) const = 0;
	virtual void solve(vec&, const vec& ) const = 0;

    Data<bool> _constant;


protected:

    bool _firstFactorization;
	
};


}
}
}

#endif
