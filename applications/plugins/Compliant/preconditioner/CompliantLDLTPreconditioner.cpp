#include "CompliantLDLTPreconditioner.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(CompliantLDLTPreconditioner);
int CompliantLDLTPreconditionerClass = core::RegisterObject("LDLT preconditioner").add< CompliantLDLTPreconditioner >();


CompliantLDLTPreconditioner::CompliantLDLTPreconditioner()
    : BasePreconditioner()
    , _factorized( false )
{

}

void CompliantLDLTPreconditioner::compute( const AssembledSystem::mat& H )
{
    if( !_factorized )
    {
        _factorized = true;

        preconditioner.compute( H.selfadjointView<Eigen::Lower>() );

        if( preconditioner.info() != Eigen::Success )
        {
            std::cerr<<SOFA_CLASS_METHOD<<"automatic regularization of a singular matrix\n";

            // if singular, try to regularize by adding a tiny diagonal matrix
            AssembledSystem::mat identity(H.rows(),H.cols());
            identity.setIdentity();
            preconditioner.compute( (H + identity * std::numeric_limits<SReal>::epsilon()).selfadjointView<Eigen::Lower>() );

            if( preconditioner.info() != Eigen::Success )
            {
                serr << "non invertible response" << sendl;
                assert( false );
            }

        }
    }
}

void CompliantLDLTPreconditioner::apply( AssembledSystem::vec& res, const AssembledSystem::vec& v )
{
    res.resize( v.size() );
    res.head(preconditioner.rows()) = preconditioner.solve( v.head(preconditioner.rows()) );
    res.tail( v.size()-preconditioner.rows() ) = v.tail( v.size()-preconditioner.rows() ); // in case of dofs have been added, like mouse...;
}

}
}
}

 
