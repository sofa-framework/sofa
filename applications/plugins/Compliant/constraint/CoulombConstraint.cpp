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

void CoulombConstraint::project( SReal* out, unsigned n, bool correctionPass ) const
{
    assert( n >= 3 );

    if( correctionPass )
    {
        // during correction pass, only the unilateral projection is performed
        // the cone projection will be performed during the dynamics pass
        for( unsigned i = 0 ; i < n ; ++i )
            out[i] = std::max( (SReal)0.0, out[i] );
    }
    else
    {
    //    std::cerr<<SOFA_CLASS_METHOD<<out[0]<<" "<<out[1]<<" "<<out[2]<<"     "<<mu<<std::endl;

       typedef Eigen::Matrix<SReal, 3, 1> vec3;
       Eigen::Map< vec3 > view(out);
       // std::cout << "before: " << view.transpose()  << " , norm = " <<  view.norm() << std::endl;

       // Coulomb Friction

       // max: this one is broken !
//        coneProjection<SReal>( out, mu );

       // reverting to slow-but-working code ;)
       vec3 normal = vec3::UnitX();
       view = cone<SReal>(view, normal, mu);

       // std::cout << "after: " << view.transpose() << " , norm = " <<  view.norm() << std::endl;
    }

}


}
}
}

