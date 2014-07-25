#ifndef FLEXIBLE_UTILS_SE3_H
#define FLEXIBLE_UTILS_SE3_H

// SE(3) kinematics

// author: maxime.tournier@inria.fr
// license: LGPL 2.1

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Quat.h>

#include <limits>

// TODO include boost::math in sofa or implement SE3::sinc more precisely
// #include <boost/math/special_functions/sinc.hpp>

template<class U>
struct SE3 {

	typedef U real;

	typedef ::sofa::defaulttype::RigidCoord<3, real> coord_type;
	typedef ::sofa::defaulttype::RigidDeriv<3, real> deriv_type;

	typedef Eigen::Matrix<real, 3, 1> vec3;
	typedef Eigen::Matrix<real, 6, 1> vec6;

	typedef Eigen::Matrix<real, 6, 6> mat66;
	typedef Eigen::Matrix<real, 3, 3> mat33;
    typedef Eigen::Matrix<real, 3, 6> mat36;

	typedef Eigen::Quaternion<real> quat;

	// order: translation, rotation
	typedef vec6 twist;


	// easy mappings between sofa/eigen vectors
	template<int I>
	static Eigen::Map< Eigen::Matrix<real, I, 1> > map(::sofa::defaulttype::Vec<I, real>& v) {
		return Eigen::Map< Eigen::Matrix<real, I, 1> >(v.ptr());
	}

	template<int I>
	static Eigen::Map<const Eigen::Matrix<real, I, 1> > map(const ::sofa::defaulttype::Vec<I, real>& v) {
		return Eigen::Map<const Eigen::Matrix<real, I, 1> >(v.ptr());
	}



	//  quaternion conversion
	static quat coord(const ::sofa::helper::Quater<real>& at) {
		return quat(at[3],
		            at[0],
		            at[1],
		            at[2]);
	}

	static ::sofa::helper::Quater<real> coord(const quat& at) {
		return ::sofa::helper::Quater<real>(at.coeffs()(1),
		                                    at.coeffs()(2),
		                                    at.coeffs()(3),
		                                    at.coeffs()(0));
	}
	
	// rotation quaternion
	static quat rotation(const coord_type& at) {
		return coord(at.getOrientation() );
	}

	
	// translation vector
	static vec3 translation(const coord_type& at) {
		return map( at.getCenter() );
	}


	// standard coordinates for SE(3) tangent vectors are body and
	// spatial coordinates. SOFA uses its own custom coordinate system,
	// so here are a couple of conversion routines.

	// sofa -> body velocity coordinates conversion
	static twist body(const coord_type& at, const deriv_type& sofa) {
		twist res;

		quat qT = rotation( at ).conjugate();
		
		// orientation
		res.template tail<3>() = qT * map(sofa.getVOrientation() );
		res.template head<3>() = qT * map(sofa.getVCenter() );

		return res;
	}


	// sofa -> body, matrix version
	static mat66 body(const coord_type& at) {
		mat66 res;

		mat33 R = rotation(at).normalized().toRotationMatrix();

		res <<
			R.transpose(), mat33::Zero(),
			mat33::Zero(), R.transpose();

		return res;
	}


	// sofa -> spatial, matrix version
	static mat66 spatial(const coord_type& at) {
		// TODO optimize !!!
		return Ad(at) * body(at);
	}

	static mat66 spatial_to_sofa(const coord_type& at) {
		// TODO optimize !!!
		coord_type i = inv(at);
		return body( i ) * Ad( i );
	}
	

	// body -> sofa velocity coordinates conversion
	static deriv_type sofa(const coord_type& at, const twist& body) {
		deriv_type res;

		quat q = rotation(at);

		map( res.getVOrientation() )  = q * body.template tail<3>();
		map( res.getVCenter() ) = q * body.template head<3>();

		return res;
	}


	// body -> sofa, matrix version
	static mat66 sofa(const coord_type& at) { 
		mat66 res;

		mat33 R = rotation( at ).toRotationMatrix();

		res <<
			R,  mat33::Zero(),
			mat33::Zero(), R;

		return res;
	};

	
	// sofa adT
	static vec6 move_wrench(const vec6& w, const vec3& delta) {
		vec6 res;

		res.template head<3>() = w.template head<3>();
		res.template tail<3>() = w.template tail<3>() + delta.cross( w.template head<3>() );

		return res;
	}


	// skew-symmetric mapping: hat(v) * x = v.cross(x)
	static mat33 hat(const vec3& v) {
		mat33 res;

		res.diagonal().setZero();

		res(0, 1) = -v.z();
		res(1, 0) = v.z();

		res(0, 2) = v.y();
		res(2, 0) = -v.y();

		res(1, 2) = -v.x();
		res(2, 1) = v.x();

		return res;
	}

	// SE(3) adjoint map
	static twist Ad(const coord_type& at, const twist& v) {

		quat q = rotation(at).normalized();
		vec3 t = translation(at);

		twist res;

		res.template tail<3>() = q * v.template tail<3>();
		res.template head<3>() = t.cross(res.template tail<3>()) + q * v.template head<3>();

		return res;
	}


	//  SO(3) adjoint map quaternion -> matrix conversion
	static mat33 Ad(const ::sofa::helper::Quater<real>& at) {
		return coord(at).toRotationMatrix();
	}
	
	// SE(3) adjoint, matrix version
	static mat66 Ad(const coord_type& at) {

		mat33 R = rotation(at).normalized().toRotationMatrix();
		mat33 T = hat( translation(at) );

		mat66 res;

		res <<
			R, T * R,
			mat33::Zero(), R;

		return res;
	}


	// SE(3) group operations
	static coord_type inv(const coord_type& g) {
		return ::sofa::defaulttype::Rigid3Types::inverse( g );
	}


	static coord_type prod(const coord_type& a, const coord_type& b) {
		return ::sofa::defaulttype::Rigid3Types::mult(a, b);
	}


	static const real epsilon() {
		return std::numeric_limits<real>::epsilon();
	}



	// SO(3) log
	static vec3 log(const quat& qq) {
		
		quat q = qq;
		q.normalize();
		
		// flip if needed
		if( q.w() < 0 ) q.coeffs() = -q.coeffs();
		
		// (half) rotation angle
		real half_theta = std::acos( q.w() );
		real theta = 2 * half_theta;
		
		if( std::abs(theta) < epsilon() ) {
			return q.vec();
		} else {
			// TODO q.vec() / sinc(theta) instead ?
			return theta * q.vec().normalized();
		}

	}


	static real sinc( const real& x ) {
		
		if( std::abs(x) < epsilon() ) {
			// TODO Taylor series similar to boost::math::sinc instead
			return 1.0;
		}

		return std::sin(x) / x;
	}


	// SO(3) log derivative, body coordinates
	static mat33 dlog(const quat& q) {
		vec3 log_q = log(q);
		mat33 res = mat33::Identity() + hat( log_q );

		real theta = log_q.norm();
		if( theta < epsilon() ) return res;

		vec3 n = log_q.normalized();

		real cos = std::cos(theta);
        
		real sinc = SE3::sinc(theta);

		assert( std::abs( sinc ) > epsilon() );

		real alpha = cos / sinc - 1.0;
		// real alpha = theta / std::tan(theta) - 1.0;

		res += alpha * (mat33::Identity() - n * n.transpose() );

		return res;
	}



	// SO(3) exp derivative, body coordinates. 
	static mat33 dexp(const vec3& x, const quat& exp_v) {
		mat33 res = mat33::Zero();

		mat33 xhat = hat(x);
		mat33 R = exp_v.toRotationMatrix();
		
		res = mat33::Identity();

		real theta2 = x.squaredNorm();

		if( theta2 > epsilon() ) {
			res.noalias() = res + (R.transpose() - mat33::Identity() + xhat) * xhat / theta2;
		}
		
		return res;
	}

	// TODO provide exponential lol
	
	// static mat33 dexp(const vec3& x) {
	// 	return dexp(x, exp(x) );
	// }


	// R(3) x SO(3) logarithm (i.e. *not* SE(3))
	static deriv_type product_log(const coord_type& g) {

		deriv_type res;

		res.getVCenter() = g.getCenter();
		map(res.getVOrientation() ) = log( rotation(g) );

		return res;
	}


	// R(3) x SO(3) logarithm derivative, in sofa coordinates
	static mat66 product_dlog(const coord_type& g) {
		mat66 res;

		quat q = rotation(g);
		mat33 R = q.toRotationMatrix();

		res <<
			mat33::Identity(), mat33::Zero(),
			mat33::Zero(), dlog( q ) * R.transpose();

		return res;
	}


	// left and right translation derivatives:

	// L_h(g) = h.g
	// R_h(g) = g.h
	
	// dL_h(g) in sofa coordinates
	static mat66 dL(const coord_type& h, const coord_type& ) {
		// dL_h(g) = sofa(hg) * body(g) = sofa(hg) * sofa(g-1) = sofa(h)
		return sofa(h);
	}


	// dR_h(g) in sofa coordinates
	static mat66 dR(const coord_type& h, const coord_type& g) {
		mat66 res; 

		res <<
			mat33::Identity(), - hat( rotation(g) * translation(h) ),
			mat33::Zero(), mat33::Identity();

		return res;

		// return sofa( prod(g, h) ) * Ad( inv(h) ) * body(g);
	}


};


#endif
