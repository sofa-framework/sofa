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
    , horizontalProjection( true )
{
	assert(mu >= 0); 
}

void CoulombConstraint::project( SReal* out, unsigned n, unsigned /*index*/, bool correct ) const
{
    assert( n % 3  == 0);    

    // max: whatever happened to the old, correct version ? rewriting.
    const unsigned m = n / 3;

    typedef Eigen::Matrix<SReal, 3, 1> vec3;
    static const vec3 normal = vec3::UnitX();

    // projection operator
    typedef vec3 (*proj_type)(const vec3&, const vec3& , SReal );
    const proj_type proj = horizontalProjection ?
        cone_horizontal<SReal> : cone<SReal>;
    
    for(unsigned i = 0; i < m; ++i) {

        Eigen::Map< vec3 > view(&out[3*i]);

        if( correct ) {
            view = proj(view, normal, 0.0);
        } else {
            view = proj(view, normal, mu);
        }

    }
    
    return;
    
    // TODO FIXME why do we only consider the first contact ?!
    
    // By construction the first component is aligned with the normal

    // no attractive force
    if( out[0] < 0 )
    {
        for( unsigned int i=0 ; i<n ; ++i ) out[i] = 0;
    }
    else
    {
         if( correct )
         {
             // only keep unilateral projection during correction
             for( unsigned int i=1 ; i<n ; ++i ) out[i] = 0;
         }
         else
         {
             // full cone projection

             typedef Eigen::Matrix<SReal, 3, 1> vec3;
             Eigen::Map< vec3 > view(out);

             static const vec3 normal = vec3::UnitX();

             // could be optimized by forcing normal=unitX in cone projection

             if( horizontalProjection )
                 view = cone_horizontal<SReal>(view, normal, mu);
             else
                // coneProjection(out, mu);
                view = cone<SReal>(view, normal, mu);

             // TODO wtf is this ?
             for( unsigned int i=3 ; i<n ; ++i ) out[i] = 0;
         }
    }

}


}
}
}

