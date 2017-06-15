/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// Author: Francois Faure, INRIA-UJF, (C) 2006

#ifndef SOFA_DEFAULTTYPE_SOLIDTYPES_H
#define SOFA_DEFAULTTYPE_SOLIDTYPES_H

#include <sofa/defaulttype/defaulttype.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <iostream>
#include <map>

#ifdef PS3
typedef float GLdouble;
#else
typedef double GLdouble;
#endif

namespace sofa
{

namespace defaulttype
{

/**
Base types for the ArticulatedSolid: position, orientation, velocity, angular velocity, etc.

@author François Faure, INRIA-UJF, 2006
*/

class SOFA_DEFAULTTYPE_API Transform;


template< class R=float >
class SOFA_DEFAULTTYPE_API SolidTypes
{
public:
    typedef R Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef Vec3 Vec;  ///< For compatibility
    typedef helper::Quater<Real> Rot;
    typedef defaulttype::Mat<3,3,Real> Mat3x3;
    typedef Mat3x3 Mat; ///< For compatibility
    typedef defaulttype::Mat<6,6,Real> Mat6x6;
    typedef Mat6x6 Mat66; ///< For compatibility
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef Vec6 DOF; ///< For compatibility


    /** A spatial vector.
    When representing a velocity, lineVec is the angular velocity and freeVec is the linear velocity.
    When representing a spatial force, lineVec is the force and freeVec is the torque. */
    class SOFA_DEFAULTTYPE_API SpatialVector
    {
    public:
        Vec lineVec;
        Vec freeVec;
        void clear();
        SpatialVector();
        /**
        \param l The line vector: angular velocity, or force
        \param f The free vector: linear velocity, or torque
        */
        SpatialVector( const Vec& l, const Vec& f );


        SpatialVector& operator += (const SpatialVector& v);

        //template<class Real2>
        SpatialVector operator * ( Real a ) const
        {
            return SpatialVector( lineVec *a, freeVec * a);
        }

        //template<class Real2>
        SpatialVector& operator *= ( Real a )
        {
            lineVec *=a;
            freeVec *= a;
            return *this;
        }

        SpatialVector operator + ( const SpatialVector& v ) const;
        SpatialVector operator - ( const SpatialVector& v ) const;
        SpatialVector operator - ( ) const;
        /// Spatial dot product (cross terms)
        Real operator * ( const SpatialVector& v ) const;
        /// Spatial cross product
        SpatialVector cross( const SpatialVector& v ) const;
        /// product with a dense matrix
        SpatialVector operator * (const Mat66&) const;

        /// write to an output stream
        inline friend std::ostream& operator << (std::ostream& out, const SpatialVector& t )
        {
            out << t.lineVec << " " << t.freeVec;
            return out;
        }

        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, SpatialVector& t )
        {
            in >> t.lineVec >> t.freeVec;
            return in;
        }

        /// If the SpatialVector models a spatial velocity, then the linear velocity is the freeVec.
        /// Otherwise, the SpatialVector models a spatial force, and this method returns a torque.
        Vec& getLinearVelocity()
        {
            return freeVec;
        }
        const Vec& getLinearVelocity() const
        {
            return freeVec;
        }
        void setLinearVelocity(const Vec& v)
        {
            freeVec = v;
        }
        /// If the SpatialVector models a spatial velocity, then the angular velocity is the lineVec.
        /// Otherwise, the SpatialVector models a spatial force, and this method returns a force.
        Vec& getAngularVelocity()
        {
            return lineVec;
        }
        const Vec& getAngularVelocity() const
        {
            return lineVec;
        }
        void setAngularVelocity(const Vec& v)
        {
            lineVec = v;
        }

        /// If the SpatialVector models a spatial force, then the torque is the freeVec.
        /// Otherwise, the SpatialVector models a spatial velocity, and this method returns a linear velocity.
        Vec& getTorque()
        {
            return freeVec;
        }
        const Vec& getTorque() const
        {
            return freeVec;
        }
        void setTorque(const Vec& v)
        {
            freeVec = v;
        }
        /// If the SpatialVector models a spatial force, then the torque is the lineVec.
        /// Otherwise, the SpatialVector models a spatial velocity, and this method returns an angular velocity.
        Vec& getForce()
        {
            return lineVec;
        }
        const Vec& getForce() const
        {
            return lineVec;
        }
        void setForce(const Vec& v)
        {
            lineVec = v;
        }
    };

    /**
     * \brief A twist aka a SpatialVector representing a velocity
     * This is pratically a SpatialVector (screw) with the additionnal semantics
     * that this screw represents a twist (velocity) and not a wrench (force and torque)
     * @author Anthony Truchet, CEA, 2006
     */
    class Twist : public SpatialVector
    {
    public:
        Twist(const Vec3& linear, const Vec3& angular)
            : SpatialVector(angular, linear) {}
    };

    /**
    	* \brief A wrench aka a SpatialVector representing a force and a torque
     * This is pratically a SpatialVector (screw) with the additionnal semantics
     * that this screw represents a wrench (force and torque) and not a twist (velocity)
     * @author Anthony Truchet, CEA, 2006
     */
    class Wrench : public SpatialVector
    {
    public:
        Wrench(const Vec3& force, const Vec3& torque)
            : SpatialVector(force, torque) {}
    };

    /** Define a frame (child) whith respect to another (parent). A frame represents a local coordinate system.

    Internal data represents the orientation of the child wrt the parent, BUT the translation vector represents the origin of the parent with respect to the child. For example, the coordinates M_p of point M in parent given the coordinates M_c of the same point in child are given by: M_p = orientation * ( M_c - origin ). This is due to Featherstone's conventions. Use method setTranslationRotation( const Vec& t, const Rot& q ) to model the Transform the standard way (i.e. translation givne in the parent frame).


    */
    class SOFA_DEFAULTTYPE_API Transform
    {
    public:
        /// The default constructor does not initialize the transform
        Transform();
        /// Origin of the child in parent coordinates, orientation of the child wrt to parent
        Transform( const Vec& origin, const Rot& orientation );
        /// WARNING: using Featherstone's conventions (see class documentation)
        Transform( const Rot& q, const Vec& o );
        /// Origin of the child in the parent coordinate system and the orientation of the child wrt the parent (i.e. standard way)
        void set( const Vec& t, const Rot& q );
        /// Reset this to identity
        void clear();
        /// The identity transform (child = parent)
        static Transform identity();
        /// Origin of the child in the parent coordinate system and the orientation of the child wrt the parent (i.e. standard way)
        //static Transform inParent(const Vec& t, const Rot& r);
        /// Define child as a given SpatialVector integrated during one second, starting from the parent (used for time integration). The spatial vector is given in parent coordinates.
        Transform( const SpatialVector& v );
        /// The inverse transform i.e. parent wrt child
        Transform inversed() const;
        /// Parent origin in child coordinates (the way it is actually stored internally)
        const Vec& getOriginOfParentInChild() const;
        /// Origin of child in parent coordinates
        Vec getOrigin() const;
        /// Origin of child in parent coordinates
        void setOrigin( const Vec& );
        /// Orientation of the child coordinate axes wrt the parent coordinate axes
        const Rot& getOrientation() const;
        /// Orientation of the child coordinate axes wrt the parent coordinate axes
        void setOrientation( const Rot& );
        /// Matrix which projects vectors from child coordinates to parent coordinates. The columns of the matrix are the axes of the child base axes in the parent coordinate system.
        Mat3x3 getRotationMatrix() const;






        /**
         * \brief Adjoint matrix to the transform
         * This matrix transports velocities in twist coordinates from the child frame to the parent frame.
         * Its inverse transpose does the same for the wrenches
         */
        Mat6x6 getAdjointMatrix() const;

        /// Project a vector (i.e. a direction or a displacement) from child coordinates to parent coordinates
        Vec projectVector( const Vec& vectorInChild ) const;
        /// Project a point from child coordinates to parent coordinates
        Vec projectPoint( const Vec& pointInChild ) const;
        /// Projected a vector (i.e. a direction or a displacement) from parent coordinates to child coordinates
        Vec backProjectVector( const Vec& vectorInParent ) const;
        /// Project point from parent coordinates to this coordinates
        Vec backProjectPoint( const Vec& pointInParent ) const;
        /// Combine two transforms. If (*this) locates frame B (child) wrt frame A (parent) and if f2 locates frame C (child) wrt frame B (parent) then the result locates frame C wrt to Frame A.
        Transform operator * (const Transform& f2) const;
        /// Combine two transforms. If (*this) locates frame B (child) wrt frame A (parent) and if f2 locates frame C (child) wrt frame B (parent) then the result locates frame C wrt to Frame A.
        Transform& operator *= (const Transform& f2);

        /** Project a spatial vector from child to parent
        	*  TODO One should handle differently the transformation of a twist and a wrench !
        	*  This applying the adjoint to velocities or its transpose to wrench :
        	*  V_parent = Ad . V_child or W_child = Ad^T . W_parent
        	*  To project a wrench in the child frame to the parent frame you need to do
        	*  parent_wrench = this->inversed * child_wrench
        	*  (this doc needs to be douv-ble checked !)
        	*/
        // create a spatial Vector from a small transformation
        SpatialVector  CreateSpatialVector();
        SpatialVector DTrans();

        SpatialVector operator * (const SpatialVector& sv ) const;
        /// Project a spatial vector from parent to child (the inverse of operator *). This method computes (*this).inversed()*sv without inverting (*this).
        SpatialVector operator / (const SpatialVector& sv ) const;
        /// Write an OpenGL matrix encoding the transformation of the coordinate system of the child wrt the coordinate system of the parent.
        void writeOpenGlMatrix( GLdouble *m ) const;
        /// Draw the axes of the child coordinate system in the parent coordinate system
        // void glDraw() const;
        /// Print the origin of the child in the parent coordinate system and the quaternion defining the orientation of the child wrt the parent
        inline friend std::ostream& operator << (std::ostream& out, const Transform& t )
        {
            // out<<"("<<t.getOrigin()<<")";
            // out<<"("<<t.getOrientation()<<")";

            out << t.getOrigin() << " " << t.getOrientation();

            return out;
        }

        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Transform& t )
        {
            Vec origin;
            Rot orientation;

            in >> origin >> orientation;

            t.set(origin, orientation);

            return in;
        }

        /// Print the internal values (i.e. using Featherstone's conventions, see class documentation)
        void printInternal( std::ostream&) const;

        /** @name Time integration
        * Methods used in time integration
        */
        ///@{
        /// (*this) *= Transform(v)  Used for time integration. SHOULD WE RATHER APPLY (*this)=Transform(v)*(*this) ???
        Transform& operator +=(const SpatialVector& a);

        Transform& operator +=(const Transform& a);

        template<class Real2>
        Transform& operator*=(Real2 a)
        {
            std::cout << "SolidTypes<R>::Transform::operator *="<<std::endl;
            origin_ *= a;
            //orientation *= a;
            return *this;
        }

        template<class Real2>
        Transform operator*(Real2 a) const
        {
            Transform r = *this;
            r*=a;
            return r;
        }
        ///@}


    protected:
        Rot orientation_; ///< child wrt parent
        Vec origin_;  ///< parent wrt child

    };


    class SOFA_DEFAULTTYPE_API RigidInertia
    {
    public:
        Real m;  ///< mass
        Vec h;   ///< position of the mass center in the local reference frame
        Mat I;  /// Inertia matrix around the mass center
        RigidInertia();
        RigidInertia( Real m, const Vec& h, const Mat& I );
        SpatialVector operator * (const SpatialVector& v ) const;
        RigidInertia operator * ( const Transform& t ) const;
        inline friend std::ostream& operator << (std::ostream& out, const RigidInertia& r )
        {
            out<<"I= "<<r.I<<std::endl;
            out<<"h= "<<r.h<<std::endl;
            out<<"m= "<<r.m<<std::endl;
            return out;
        }
    };

    class SOFA_DEFAULTTYPE_API ArticulatedInertia
    {
    public:
        Mat M;
        Mat H;
        Mat I;
        ArticulatedInertia();
        ArticulatedInertia( const Mat& M, const Mat& H, const Mat& I );
        SpatialVector operator * (const SpatialVector& v ) const;
        ArticulatedInertia operator * ( Real r ) const;
        ArticulatedInertia& operator = (const RigidInertia& Ri );
        ArticulatedInertia& operator += (const ArticulatedInertia& Ai );
        ArticulatedInertia operator + (const ArticulatedInertia& Ai ) const;
        ArticulatedInertia operator - (const ArticulatedInertia& Ai ) const;
        inline friend std::ostream& operator << (std::ostream& out, const ArticulatedInertia& r )
        {
            out<<"I= "<<r.I<<std::endl;
            out<<"H= "<<r.H<<std::endl;
            out<<"M= "<<r.M<<std::endl;
            return out;
        }
        /// Convert to a full 6x6 matrix
        void copyTo( Mat66& ) const;
    };

    typedef Transform Coord;
    typedef SpatialVector Deriv;
    typedef Coord VecCoord;
    typedef Deriv VecDeriv;
    typedef Real  VecReal;

    static Mat dyad( const Vec& u, const Vec& v );

    static Vec mult( const Mat& m, const Vec& v );

    static Vec multTrans( const Mat& m, const Vec& v );

    /// Cross product matrix of a vector
    static Mat crossM( const Vec& v );

    static ArticulatedInertia dyad ( const SpatialVector& u, const SpatialVector& v );

    static const char* Name()
    {
        return "Solid";
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_DEFAULTTYPE_SOLIDTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_DEFAULTTYPE_API SolidTypes<double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_DEFAULTTYPE_API SolidTypes<float>;
#endif
#endif

}// defaulttype

}// sofa



#endif


