#include "UnilateralConstraint.h"
#include <sofa/core/ObjectFactory.h>

//#include <Eigen/Core>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(UnilateralConstraint);
int UnilateralConstraintClass = core::RegisterObject("Unilateral constraint")
        .add< UnilateralConstraint >()
        .addAlias("UnilateralProjector"); // eheh :p

void UnilateralConstraint::project(SReal* out, unsigned n) const {
    //Eigen::Map< Eigen::Matrix<SReal, Eigen::Dynamic, 1> > view(out, n);

	// std::cerr << "before: " << view.transpose() << std::endl;
	for(unsigned i = 0; i < n; ++i) { 
        out[i] = std::max(0.0, out[i]);
	}
	// std::cerr << "after: " << view.transpose() << std::endl;
	
}

}
}
}
