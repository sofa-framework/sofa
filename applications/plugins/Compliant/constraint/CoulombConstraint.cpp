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
    assert( n >= 3 );

	typedef Eigen::Matrix<SReal, 3, 1> vec3;
	Eigen::Map< vec3 > view(out);

	static const vec3 normal = vec3::UnitX();

	// only project normal component during correction
    if( correct ) {
		SReal direction = normal.dot( view );

		if( direction < 0 ) {
			// only project normal
			view = view - normal * normal.dot( view );

            // un bazooka pour tuer une mouche
            // ca ne revient pas exactement à :
            // if( view[O] < 0 ) view[O] = 0;
            // ?
            // et quid des forces tangentielles ?
            // il ne faut pas les "unilatéraliser" aussi ?
            // pour les empecher d'attirer.
		}
		
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

