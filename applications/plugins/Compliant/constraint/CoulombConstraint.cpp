#include "CoulombConstraint.h"

//#include <Eigen/Core>

//#include "./utils/nan.h"
#include "../utils/cone.h"
//#include "./utils/map.h"

namespace sofa {
namespace component {
namespace linearsolver {


CoulombConstraint::CoulombConstraint(SReal mu)
	: mu(mu) { 
	assert(mu >= 0); 
}

void CoulombConstraint::project( SReal* out, unsigned n) const
{
    assert( n >= 3 );

	// side-node: this is how you turn off an 'unused variable'
	// warning: 
	(void)n;

//    std::cerr<<SOFA_CLASS_METHOD<<out[0]<<" "<<out[1]<<" "<<out[2]<<"     "<<mu<<std::endl;

   typedef Eigen::Matrix<SReal, 3, 1> vec3;
   Eigen::Map< vec3 > view(out);
   // std::cout << "before: " << view.transpose()  << " , norm = " <<  view.norm() << std::endl;

   // Coulomb Friction

   // max: this one is broken !
   coneProjection<SReal>( out, mu );

   // reverting to slow-but-working code ;)
   // vec3 normal = vec3::UnitX(); 
   // view = cone<SReal>(view, normal, mu);

   // std::cout << "after: " << view.transpose() << " , norm = " <<  view.norm() << std::endl;

}


}
}
}

