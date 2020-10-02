#include "CoulombConstraint.h"
#include "../utils/cone.h"

namespace sofa {
namespace component {
namespace linearsolver {

template<class DataTypes>
CoulombConstraint<DataTypes>::CoulombConstraint(SReal mu)
    : horizontalProjection( true )
{
    this->mu = mu;
	assert(mu >= 0); 
}

template<class DataTypes>
void CoulombConstraint<DataTypes>::project( SReal* out, unsigned n, unsigned /*index*/, bool correct ) const
{
    enum {N = DataTypes::deriv_total_size };

    assert( n % N  == 0);

    typedef Eigen::Matrix<SReal, 3, 1> vec3;
    typedef Eigen::Map< vec3 > view_type;    
    
    // contact count
    const unsigned m = n / N;
    
    static const vec3 normal = vec3::UnitX();

    // projection operator
    typedef vec3 (*proj_type)(const vec3&, const vec3& , SReal );
    
    const proj_type proj = horizontalProjection ?
        cone_horizontal<SReal> : cone<SReal>;

    // correction: degenerate (line) cone
    const SReal gamma = correct ? 0.0 : mu;
    
    for(unsigned i = 0; i < m; ++i) {

        view_type view(&out[ N*i]);
        view = proj(view, normal, gamma);
        
        // zero remaining coordinates (e.g. for rigids)
        // TODO even when not correcting ?
        for(unsigned j = 3; j < N; ++j) {
            out[ N*i + j ] = 0;
        }
    }
    
    return;
    
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

