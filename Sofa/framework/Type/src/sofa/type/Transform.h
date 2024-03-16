/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/type/SpatialVector.h>
#include <sofa/type/Quat.h>

namespace sofa::type
{

/**
 * Define a frame (child) whith respect to another (parent). A frame represents a local coordinate system.
 *
 * Internal data represents the orientation of the child wrt the parent, BUT the
 * translation vector represents the origin of the parent with respect to the
 * child. For example, the coordinates M_p of point M in parent given the
 * coordinates M_c of the same point in child are given by:
 * M_p = orientation * ( M_c - origin ). This is due to Featherstone's
 * conventions. Use method setTranslationRotation( const Vec& t, const Rot& q )
 * to model the Transform the standard way (i.e. translation givne in the parent frame).
 **/
template<class TReal>
class Transform
{
public:
    using Real = TReal;
    using Vec = sofa::type::Vec<3, TReal>;
    using Rot = type::Quat<Real>;
    using Mat3x3 = type::Mat<3,3,Real>;
    using Mat6x6 = sofa::type::Mat<6, 6, TReal>;

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
    /// Define child as a given SpatialVector<TReal> integrated during one second, starting from the parent (used for time integration). The spatial vector is given in parent coordinates.
    Transform( const SpatialVector<TReal>& v );
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
    SpatialVector<TReal>  CreateSpatialVector();
    SpatialVector<TReal> DTrans();

    SpatialVector<TReal> operator * (const SpatialVector<TReal>& sv ) const;
    /// Project a spatial vector from parent to child (the inverse of operator *). This method computes (*this).inversed()*sv without inverting (*this).
    SpatialVector<TReal> operator / (const SpatialVector<TReal>& sv ) const;
    /// Write an OpenGL matrix encoding the transformation of the coordinate system of the child wrt the coordinate system of the parent.
    void writeOpenGlMatrix( double *m ) const;
    /// Draw the axes of the child coordinate system in the parent coordinate system
    /// Print the origin of the child in the parent coordinate system and the quaternion defining the orientation of the child wrt the parent
    inline friend std::ostream& operator << (std::ostream& out, const Transform& t )
    {
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
    Transform& operator +=(const SpatialVector<TReal>& a);

    Transform& operator +=(const Transform& a);

    template<class Real2>
    Transform& operator*=(Real2 a)
    {
        origin_ *= a;
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

#if !defined(SOFA_TYPE_TRANSFORM_CPP)
extern template class SOFA_TYPE_API Transform<double>;
extern template class SOFA_TYPE_API Transform<float>;
#endif

}
