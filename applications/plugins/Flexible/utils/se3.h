#ifndef FLEXIBLE_UTILS_SE3_H
#define FLEXIBLE_UTILS_SE3_H

// SE(3) kinematics.

// author: maxime.tournier@inria.fr
// license: LGPL 2.1

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sofa/defaulttype/RigidTypes.h>
#include <boost/math/special_functions/sinc.hpp>

template<class U>
struct SE3
{

    typedef U real;

    // TO DO SOFA types
    typedef ::sofa::defaulttype::RigidCoord<3, real> coord_type;
    typedef ::sofa::defaulttype::RigidDeriv<3, real> deriv_type;

    typedef Eigen::Matrix<real, 3, 1> vec3;
    typedef Eigen::Matrix<real, 6, 1> vec6;

    typedef Eigen::Matrix<real, 6, 6> mat66;
    typedef Eigen::Matrix<real, 3, 3> mat33;

    typedef Eigen::Quaternion<real> quat;

    // order: translation, rotation
    typedef vec6 twist;


    // easy mapping between vector types
    static Eigen::Map<vec3> map(::sofa::defaulttype::Vec<3, real>& v)
    {
        return Eigen::Map<vec3>(v.ptr());
    }

    static Eigen::Map<const vec3> map(const ::sofa::defaulttype::Vec<3, real>& v)
    {
        return Eigen::Map<const vec3>(v.ptr());
    }



    // rotation quaternion
    static quat rotation(const coord_type& at)
    {
        return quat(at.getOrientation()[3],
                at.getOrientation()[0],
                at.getOrientation()[1],
                at.getOrientation()[2]);
    }

    // translation vector
    static vec3 translation(const coord_type& at)
    {
        return map( at.getCenter() );
    }

    // sofa -> body velocity conversion
    static twist body(const coord_type& at, const deriv_type& sofa)
    {
        twist res;

        quat qT = rotation(at).conjugate();

        // orientation
        res.template tail<3>() = qT * map(sofa.getVOrientation() );
        res.template head<3>() = qT * map(sofa.getVCenter() );

        return res;
    }

    // sofa -> body, matrix version
    static mat66 body(const coord_type& at)
    {
        mat66 res;

        mat33 R = rotation(at).toRotationMatrix();

        res <<
            R.transpose(), mat33::Zero(),
                        mat33::Zero(), R.transpose();

        return res;
    }


    // body -> sofa velocity conversion
    static deriv_type sofa(const coord_type& at, const twist& body)
    {
        deriv_type res;

        quat q = rotation(at);

        map( res.getVOrientation() )  = q * body.template tail<3>();
        map( res.getVCenter() ) = q * body.template head<3>();

        return res;
    }

    // body -> sofa, matrix version
    static mat66 sofa(const coord_type& at)
    {
        mat66 res;

        mat33 R = rotation(at).toRotationMatrix();

        res <<
            R,  mat33::Zero(),
                mat33::Zero(), R;

        return res;
    };


    // SE(3) adjoint
    static twist ad(const coord_type& at, const twist& v)
    {

        quat q = rotation(at);
        vec3 t = translation(at);

        twist res;

        res.template tail<3>() = q * v.template tail<3>();

        res.template head<3>() = t.cross(res.template tail<3>()) + q * v.template head<3>();

        return res;
    }

    // skew-symmetric mapping
    static mat33 hat(const vec3& v)
    {
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

    // matrix version
    static mat66 ad(const coord_type& at)
    {

        mat33 R = rotation(at).toRotationMatrix();
        mat33 T = hat( translation(at) );

        mat66 res;
        res <<
            R, T * R,
               mat33::Zero(), R;

        return res;
    }

    // group operations
    static coord_type inv(const coord_type& g)
    {
        return ::sofa::defaulttype::Rigid3Types::inverse( g );
    }

    static coord_type prod(const coord_type& a, const coord_type& b)
    {
        return ::sofa::defaulttype::Rigid3Types::mult(a, b);
    }


    static const real epsilon = 1e-7;



    // SO(3) log
    static vec3 log(quat q)
    {

        q.normalize();

        // flip if needed
        if( q.w() < 0 ) q.coeffs() = -q.coeffs();

        // (half) rotation angle
        real theta = std::asin( q.vec().norm() );

        if( std::abs(theta) < epsilon )
        {
            return q.vec();
        }
        else
        {
            // TODO use boost::sinc ?
            return theta * q.vec().normalized();
        }

    }

    // SO(3) log body differential
    static mat33 dlog(const quat& q)
    {

        vec3 log_q = log(q);
        real theta = log_q.norm();

        if( theta < epsilon ) return mat33::Identity();

        vec3 n = log_q.normalized();

        real cos = std::cos(theta);
        real sinc = boost::math::sinc_pi(theta);

        real alpha = cos / sinc;

        mat33 res = mat33::Zero();

        res += alpha * mat33::Identity();
        res += hat( log_q );

        res -= (alpha - 1) * n * n.transpose();

        return res;
    }


    // R(3) x SO(3) logarithm (i.e. *not* SE(3))
    static deriv_type product_log(const coord_type& g)
    {

        deriv_type res;

        res.getVCenter() = g.getCenter();
        map(res.getVOrientation() ) = log( rotation(g) );

        return res;
    }

    // R(3) x SO(3) logarithm derivative. applies to *sofa* velocities !
    static mat66 product_dlog(const coord_type& g)
    {
        mat66 res;

        quat q = rotation(g);
        mat33 R = q.toRotationMatrix();

        res <<
            mat33::Identity(), mat33::Zero(),
                  mat33::Zero(), dlog( q ) * R.transpose();

        return res;
    }

};


#endif
