#include "LDLTPreconditioner.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(LDLTPreconditioner);
int LDLTPreconditionerClass = core::RegisterObject("LDLT preconditioner").add< LDLTPreconditioner >();


LDLTPreconditioner::LDLTPreconditioner()
    : BasePreconditioner()
    , _factorized( false )
{

}

void LDLTPreconditioner::compute( const AssembledSystem::mat& H )
{
    if( !_factorized )
    {
//         std::cerr<<SOFA_CLASS_METHOD<<"\n";

        _factorized = true;

        preconditioner.compute( H );

        if( preconditioner.info() != Eigen::Success )
        {
            // if singular, try to regularize by adding a tiny diagonal matrix
            AssembledSystem::mat identity(H.rows(),H.cols());
            identity.setIdentity();
            preconditioner.compute( H + identity * std::numeric_limits<SReal>::epsilon() );

            if( preconditioner.info() != Eigen::Success )
            {
                std::cerr << "warning: non invertible response" << std::endl;
                assert( false );
            }

        }
    }
}

void LDLTPreconditioner::apply( AssembledSystem::vec& res, const AssembledSystem::vec& v )
{
    res.resize( v.size() );
    res.head(preconditioner.rows()) = preconditioner.solve( v.head(preconditioner.rows()) );
    res.tail( v.size()-preconditioner.rows() ) = v.tail( v.size()-preconditioner.rows() ); // in case of dofs have been added, like mouse...

//    std::cerr<<SOFA_CLASS_METHOD<<"\n";
}

}
}
}

 
