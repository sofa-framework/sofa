#ifndef FLEXIBLE_UTILS_SE3_H
#define FLEXIBLE_UTILS_SE3_H

// SE(3) kinematics.

// author: maxime.tournier@inria.fr
// license: LGPL 2.1

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sofa/defaulttype/RigidTypes.h>

template<class U>
struct SE3
{

    typedef U real;

    // TODO SOFA types
    typedef ::sofa::defaulttype::RigidCoord<3, real> coord_type;
    typedef ::sofa::defaulttype::RigidDeriv<3, real> deriv_type;

    typedef Eigen::Matrix<real, 3, 1> vec3;
    typedef Eigen::Matrix<real, 6, 1> vec6;

    typedef Eigen::Matrix<real, 6, 6> mat66;
    typedef Eigen::Matrix<real, 3, 3> mat33;

    typedef Eigen::Quaternion<real> quat;

    // order: orientation, translation
    typedef vec6 twist;


    // rotation quaternion
    static quat rotation(const coord_type& at)
    {
        return quat(at.getOrientation()[3],
                at.getOrientation()[0],
                at.getOrientation()[1],
                at.getOrientation()[2]);
    }

    // rotation quaternion
    static vec3 translation(const coord_type& at)
    {
        return Eigen::Map<const vec3>(at.getCenter().ptr());
    }

    // sofa -> body velocity conversion
    static twist body(const coord_type& at, const deriv_type& sofa)
    {
        twist res;

        quat qT = rotation(at).conjugate();

        // orientation
        res.template head<3>() = qT * Eigen::Map<const vec3>(sofa.getVOrientation().ptr() );
        res.template tail<3>() = qT * Eigen::Map<const vec3>(sofa.getVCenter().ptr() );

        return res;
    }

    // sofa -> body, matrix version
    static mat66 body(const coord_type& at)
    {
        mat66 res;

        mat33 R = rotation(at).toRotationMatrix();

        res <<
            mat33::Zero(), R.transpose(),
                  R.transpose(), mat33::Zero();

        return res;
    }


    // body -> sofa velocity conversion
    static deriv_type sofa(const coord_type& at, const twist& body)
    {
        deriv_type res;

        quat q = rotation(at);


        Eigen::Map<vec3>(res.getVOrientation().ptr()) = q * body.template head<3>();
        Eigen::Map<vec3>(res.getVCenter().ptr()) = q * body.template tail<3>();

        return res;
    }

    // body -> sofa, matrix version
    static mat66 sofa(const coord_type& at)
    {
        mat66 res;

        mat33 R = rotation(at).toRotationMatrix();

        res <<
            mat33::Zero(), R,
                  R, mat33::Zero();

        return res;
    };


    // SE(3) adjoint
    static twist ad(const coord_type& at, const twist& v)
    {

        quat q = rotation(at);
        vec3 t = translation(at);

        twist res;

        res.template head<3>() = q * v.template head<3>();
        res.template tail<3>() = t.cross(res.template head<3>()) + q * v.template tail<3>();

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
            R, mat33::Zero(),
               T * R, R;

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

};


#endif
