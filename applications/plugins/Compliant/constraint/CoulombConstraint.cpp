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

void CoulombConstraint::project( SReal* out, unsigned n, bool correct ) const
{
    (void)n;
    assert( n >= 3 );

	typedef Eigen::Matrix<SReal, 3, 1> vec3;
	Eigen::Map< vec3 > view(out);

	static const vec3 normal = vec3::UnitX();

	// only keep unilateral projection during correction
    if( correct ) {
		SReal alpha = normal.dot( view );

		alpha = std::max(alpha, 0.0);

		view = normal * alpha;
		
    } else {
		// full cone projection

		// coneProjection(out, mu);
		view = cone<SReal>(view, normal, mu);
		// view = cone_horizontal<SReal>(view, normal, mu);
    }

}


}
}
}

