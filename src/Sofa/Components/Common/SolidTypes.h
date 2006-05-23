// C++ Interface: SolidTypes
//
// Description: Base types for coordinate systems and articulated bodies
//
//
// Author: Francois Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef Sofa_ComponentsSolidTypes_h
#define Sofa_ComponentsSolidTypes_h

#include <Sofa/Components/Common/Vec.h>
#include <Sofa/Components/Common/Quat.h>
#include <Sofa/Components/Common/Mat.h>
#include <Sofa/Components/Common/fixed_array.h>
#include <Sofa/Components/Common/vector.h>
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Common
{

/**
Base types for the ArticulatedSolid: position, orientation, velocity, angular velocity, etc.

@author François Faure, INRIA-UJF, 2006
*/
template< class R=float >
class SolidTypes
{
public:
    typedef R Real;
    typedef Common::Vec<3,Real> Vec;
    typedef Common::Quater<Real> Rot;
    typedef Common::Mat<3,3,Real> Mat;
    typedef Common::Vec<6,Real> DOF;



    /** A spatial vector.
    When representing a velocity, lineVec is the angular velocity and freeVec is the linear velocity.
    When representing a spatial force, lineVec is the force and freeVec is the torque. */
    class SpatialVector
    {
    public:
        Vec lineVec;
        Vec freeVec;
        void clear();
        SpatialVector();
        SpatialVector( const Vec& l, const Vec& f );
        SpatialVector& operator += (const SpatialVector& v);

        template<class Real2>
        SpatialVector operator * ( Real2 a ) const
        {
            return SpatialVector( lineVec *a, freeVec * a);
        }

        template<class Real2>
        SpatialVector& operator *= ( Real2 a )
        {
            lineVec *=a; freeVec *= a;
            return *this;
        }

        SpatialVector operator + ( const SpatialVector& v ) const;
        SpatialVector operator - ( const SpatialVector& v ) const;
        SpatialVector operator - ( ) const;
        /// Spatial dot product (cross terms)
        Real operator * ( const SpatialVector& v ) const;
        /// Spatial cross product
        SpatialVector cross( const SpatialVector& v ) const;
        inline friend std::ostream& operator << (std::ostream& out, const SpatialVector& t )
        {
            out<<t.lineVec<<" "<<t.freeVec;
            return out;
        }
    };

    /** Define a frame (this) whith respect to another (called this->parent int he documentation, although this object has no pointer to a parent Frame). A frame represents a local coordinate system.

    Internal data represents the orientation of the child wrt the parent, BUT the translation vector represents the origin of the parent with respect to the child. For example, the coordinates M_p of point M in parent given the coordinates M_c of the same point in child are given by: M_p = orientation * ( M_c - origin ). This is due to Featherstone's conventions. Use method setTranslationRotation( const Vec& t, const Rot& q ) to model the Transform the standard way (i.e. translation givne in the parent frame).


    */
    class Transform
    {
    public:
        /// The default constructor does not initialize the transform
        Transform();
        /// Define using Featherstone's conventions (see class documentation)
        Transform( const Rot& q, const Vec& o );
        /// Define given the origin of this wrt its parent and the orientation of this wrt its parent (i.e. standard way)
        void setTranslationRotation( const Vec& t, const Rot& q );
        /// Reset this to identity
        void clear();
        /// The identity transform
        static Transform identity();
        /// Define this as a given SpatialVector integrated during one second (used for time integration)
        Transform( const SpatialVector& v );
        /// The inverse transform i.e. from this->parent to this
        Transform inversed() const;
        /// Parent origin in this coordinates
        const Vec& getOriginInChild() const;
        /// Origin of this in parent coordinates
        Vec getOriginInParent() const;
        /// Set the origin, defined in parent coordinate system
        void setOriginInParent( const Vec& );
        /// Operator which projects vectors from this coordinates to parent coordinates
        const Rot& getOrientation() const;
        /// Set the orientation (this wrt parent)
        void setOrientation( const Rot& );
        /// Matrix which projects vectors from this coordinates to parent coordinates
        Mat getRotationMatrix() const;
        /// Vector projected from this coordinates to parent coordinates
        Vec projectVector( const Vec& ) const;
        /// Point projected from this coordinates to parent coordinates
        Vec projectPoint( const Vec& ) const;
        /// Vector projected from parent coordinates to this coordinates
        Vec backProjectVector( const Vec& ) const;
        /// Point projected from parent coordinates to this coordinates
        Vec backProjectPoint( const Vec& ) const;
        /// (*this)*f2 i.e. the operator to project from the f2-child of this to the parent of this
        Transform operator * (const Transform& f2) const;
        /// (*this)=(*this)*f2
        Transform& operator *= (const Transform& f2);
        /// Project a spatial vector from this to its parent coordinate system
        SpatialVector operator * (const SpatialVector& sv ) const;
        /// Project a spatial vector from its parent to this coordinate system (the inverse of operator *)
        SpatialVector operator / (const SpatialVector& sv ) const;
        /// Write this as an OpenGL matrix
        void writeOpenGlMatrix( Real *m ) const;
        /// Draw frames axes in local frame
        void glDraw() const;
        /// Print the values in the standard way (see class documentation)
        inline friend std::ostream& operator << (std::ostream& out, const Transform& t )
        {
            out<<"("<<-t.projectVector(t.origin_)<<")";
            out<<"("<<t.orientation_<<")";
            return out;
        }
        /// Print the internal values (i.e. using Featherstone's conventions)
        void printInternal( std::ostream&) const;

        /** @name Time integration
        * Methods used in time integration
        */
        //@{
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
        //@}

    protected:
        Rot orientation_; ///< child wrt parent
        Vec origin_;  ///< parent wrt child

    };


    class RigidInertia
    {
    public:
        Real m;
        Vec h;
        Mat I;
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

    class ArticulatedInertia
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
    };

    typedef Transform Coord;
    typedef SpatialVector Deriv;
    typedef std::vector<Coord> VecCoord;
    typedef std::vector<SpatialVector> VecDeriv;


    static Mat dyad( const Vec& u, const Vec& v );

    static Vec mult( const Mat& m, const Vec& v );

    static Vec multTrans( const Mat& m, const Vec& v );

    /// Cross product matrix of a vector
    static Mat crossM( const Vec& v );

    static ArticulatedInertia dyad ( const SpatialVector& u, const SpatialVector& v );
};

}//Common

}//Components

}//Sofa



#endif

