#ifndef CONE_H
#define CONE_H

#include <Eigen/Core>
#include "nan.h"

template<class U>
Eigen::Matrix<U, 3, 1> cone(const Eigen::Matrix<U, 3, 1>& f,
                            const Eigen::Matrix<U, 3, 1>& normal,
                            U mu = 1.0) {
	typedef Eigen::Matrix<U, 3, 1> vec3;

	assert( std::abs(normal.norm() - 1) < std::numeric_limits<U>::epsilon() );
	
	// normal proj
	const U theta_n = f.dot(normal); 
	
	// normal / tangent forces
	const vec3 f_n = normal * theta_n;
	const vec3 f_t = f - f_n;
	
	// tangent norm
	const U theta_t = f_t.norm();

	// warning: this includes the negative cone as well
	const bool inside_cone = (theta_t <= mu * std::abs(theta_n));
	
	// projection
	if( !inside_cone ) {

	  assert( theta_t > 0 || "impossibru !");
	  
	  // note: theta_t > 0
	  const U sign = theta_n > 0 ? 1 : -1;

	  // find positive cone generator along f_t
	  const vec3 gen = sign * f_n  +  f_t * ( std::abs(theta_n) * mu / theta_t);
	  
	  // project f onto generator
	  const U beta = gen.dot(f);
	  
	  // cone clamp
	  if( beta > 0 ) {
		const vec3 res = gen * (beta / gen.squaredNorm());
		assert( !has_nan(res) );
		return res;
	  } else {
		return vec3::Zero();
	  }		
	} else if( theta_n < 0 ) {
	  // negative cone: zero
	  return vec3::Zero();
	} else {
	  return f;
	}
}


template<class U>
Eigen::Matrix<U, 3, 1> cone_horizontal(const Eigen::Matrix<U, 3, 1>& f,
									   const Eigen::Matrix<U, 3, 1>& normal,
									   U mu = 1.0) {
	typedef Eigen::Matrix<U, 3, 1> vec3;
	
	assert( std::abs(normal.norm() - 1) < std::numeric_limits<U>::epsilon() );
	
	// normal norm
	const U theta_n = f.dot(normal); 

	// normal / tangent forces
	const vec3 f_n = normal * theta_n;
	const vec3 f_t = f - f_n;

	// tangent norm
	const U theta_t = f_t.norm();
	
	const bool inside_cone = (theta_n >= 0) && (theta_t <= mu * theta_n);
	
	// projection
	if( !inside_cone ) {
		if( theta_n <= 0 ) return vec3::Zero();
		
		return f_n + f_t * (mu * theta_n / theta_t);
	} else {
		return f;
	}
}





// with normal=(1,0,0), so f[0] is aligned with the normal
// coulomb law => normT<=mu*f[0]
// efficient cone projection from "A Matrix-free cone complementarity approach for solving large-scale, nonsmooth, rigid body dynamics", Tasora & Anitescu, 2011
template<class Real>
void coneProjection( Real* f, Real mu )
{
    // removing attracting force
    if( f[0] < 0 )
    {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        return;
    }

    const Real normT = hypot( f[1], f[2] ); // norm( f_t )

    // max: actually here it's "outside dual cone"
    // inside polar cone -> project to cone origin
    if( normT * mu > f[0] )
    {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        return;
    }

    // tangential force is very small, let's consider it as an unilateral contact
    if( normT < std::numeric_limits<Real>::epsilon() )
    {
        f[0] = std::max( (Real)0.0, f[0] );
        f[1] = 0.0;
        f[2] = 0.0;
        return;
    }

    const Real normMax = mu*f[0]; // max tangential norm to stay in the cone

    if( normT > normMax ) // outside the cone -> project
    {
        Real ratio = normMax / normT;
        f[0] = ( mu*normT + f[0] ) / ( mu*mu + 1 );
        f[1] = ratio * f[1];
        f[2] = ratio * f[2];
        return;
    }
    // else return; // inside the cone -> nothing to do
}

#endif
