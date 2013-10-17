#include "CoulombConstraint.h"

//#include <Eigen/Core>

//#include "./utils/nan.h"
#include "../utils/cone.h"
//#include "./utils/map.h"

namespace sofa {
namespace component {
namespace linearsolver {


CoulombConstraint::CoulombConstraint(SReal mu)
	: mu(mu)
{ 
	assert(mu >= 0); 
}

void CoulombConstraint::project( SReal* out, unsigned
#ifndef NDEBUG
    n
#endif
                     ) const
{

    assert( n >= 3 );


//    std::cerr<<SOFA_CLASS_METHOD<<out[0]<<" "<<out[1]<<" "<<out[2]<<"     "<<mu<<std::endl;

//    typedef Eigen::Matrix<SReal, 3, 1> vec3;
//    Eigen::Map< vec3 > view(out);
    //   std::cout << "before: " << view.transpose() << std::endl;

    // Coulomb Friction
    coneProjection<SReal>( out, mu );


    // std::cout << "projected: " << view.transpose() << std::endl;

}


}
}
}

