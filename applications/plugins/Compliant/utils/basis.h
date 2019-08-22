#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <stdexcept>

template<class U>
static inline Eigen::Matrix<U, 3, 2> ker(const Eigen::Matrix<U, 3, 1>& n) {
	typedef Eigen::Matrix<U, 3, 1> vec3;
	
	vec3 u;
	
    static const U eps = std::numeric_limits<U>::epsilon();

	if( std::abs(n(1)) > eps  ) u = vec3(0, -n(2), n(1) ).normalized();
	else if( std::abs(n(2)) > eps  ) u =  vec3(n(2), 0, -n(0) ).normalized();
	else if( std::abs(n(0)) > eps  ) u =  vec3(n(1), -n(0), 0 ).normalized();
    else throw std::logic_error("Compliant::utils::basis.h - ker - null normal");
	
	Eigen::Matrix<U, 3, 2> res;
	
	res << u, n.cross(u);
	
	return res;
}

// make an orthonormal basis for normal/tangent spaces, given normal
// vector, assumes n is normalized !
template<class U>
static inline Eigen::Matrix<U, 3, 3> basis(const Eigen::Matrix<U, 3, 1>& n) {

	Eigen::Matrix<U, 3, 3> res;
	res << n, ker(n);
	
	return res;
}


#endif
