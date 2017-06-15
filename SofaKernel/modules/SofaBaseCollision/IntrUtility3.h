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
// File modified from GeometricTools
// http://www.geometrictools.com/


#ifndef WM5INTRUTILITY3_H
#define WM5INTRUTILITY3_H
#include "config.h"

#include <sofa/defaulttype/Vec.h>
#include <SofaBaseCollision/OBBModel.h>


namespace sofa{
namespace component{
namespace collision{

template <class TReal>
struct MyBox{
    defaulttype::Vec<3,TReal> Extent;
    defaulttype::Vec<3,TReal> Axis[3];
    defaulttype::Vec<3,TReal> Center;

    void showVertices()const;
};

//----------------------------------------------------------------------------
/**
  *An IntrConfiguration is associated to a primitive projected on an axis.
  *It contains the projected interval and the order of the primitive vertices.
  */
template <typename Real>
class IntrConfiguration
{
public:
    // ContactSide (order of the intervals of projection).
    enum
    {
        LEFT,
        RIGHT,
        NONE
    };

    // VertexProjectionMap (how the vertices are projected to the minimum
    // and maximum points of the interval).
    enum
    {
        m2, m11,             // segments
        m3, m21, m12, m111,  // triangles
        m44, m2_2, m1_1      // boxes
    };

    // The VertexProjectionMap value for the configuration.
    int mMap;

    // The order of the vertices.
    int mIndex[8];

    // Projection interval.
    Real mMin, mMax;

    IntrConfiguration & operator=(const IntrConfiguration & other);
};
//----------------------------------------------------------------------------

/**
  *IntrConfiguration for capsule.
  */
template <typename Real>
class CapIntrConfiguration : public IntrConfiguration<Real>{
public:
    bool have_naxis;
    defaulttype::Vec<3,Real> axis;

    CapIntrConfiguration();

    defaulttype::Vec<3,Real> leftContactPoint(const defaulttype::Vec<3,Real> * seg,Real radius)const;
    defaulttype::Vec<3,Real> rightContactPoint(const defaulttype::Vec<3,Real> * seg,Real radius)const;

    void leftSegment(const defaulttype::Vec<3,Real> * seg,Real radius,defaulttype::Vec<3,Real> * lseg)const;
    void rightSegment(const defaulttype::Vec<3,Real> * seg,Real radius,defaulttype::Vec<3,Real> * lseg)const;

    CapIntrConfiguration & operator=(const CapIntrConfiguration & other);
};

template <class DataType>
struct IntrUtil;

template <typename Real>
struct IntrUtil{
public:
    inline static Real ZERO_TOLERANCE(){return (Real)(1e-6);}
    inline static Real SQ_ZERO_TOLERANCE(){return ZERO_TOLERANCE() * ZERO_TOLERANCE();}

    inline static void normalize(defaulttype::Vec<3,Real> & vec){
        Real n2 = vec.norm2();

        if(n2 < 1- SQ_ZERO_TOLERANCE() || n2 > 1 + SQ_ZERO_TOLERANCE())
            vec.normalize();
    }

    inline static bool normalized(const defaulttype::Vec<3,Real> & vec){
        Real n2 = vec.norm2();

        return n2 < 1 - SQ_ZERO_TOLERANCE() || n2 > 1 + SQ_ZERO_TOLERANCE();
    }

    static void ColinearSegments (const defaulttype::Vec<3,Real> segment0[2],
        const defaulttype::Vec<3,Real> segment1[2], int& quantity, defaulttype::Vec<3,Real>* P);

    static void SegmentThroughPlane (const defaulttype::Vec<3,Real> segment[2],
        const defaulttype::Vec<3,Real>& planeOrigin, const defaulttype::Vec<3,Real>& planeNormal,
        int& quantity, defaulttype::Vec<3,Real>* P);

    static void SegmentSegment (const defaulttype::Vec<3,Real> segment0[2],
        const defaulttype::Vec<3,Real> segment1[2], int& quantity, defaulttype::Vec<3,Real>* P);

    static void ColinearSegmentTriangle (const defaulttype::Vec<3,Real> segment[2],
        const defaulttype::Vec<3,Real> triangle[3], int& quantity, defaulttype::Vec<3,Real>* P);

    static void CoplanarSegmentRectangle (const defaulttype::Vec<3,Real> segment[2],
        const defaulttype::Vec<3,Real> rectangle[4], int& quantity, defaulttype::Vec<3,Real>* P);

    static void CoplanarTriangleRectangle (const defaulttype::Vec<3,Real> triangle[3],
        const defaulttype::Vec<3,Real> rectangle[4], int& quantity, defaulttype::Vec<3,Real>* P);

    static void CoplanarRectangleRectangle (
        const defaulttype::Vec<3,Real> rectangle0[4],
        const defaulttype::Vec<3,Real> rectangle1[4], int& quantity, defaulttype::Vec<3,Real>* P);

    static void projectIntPoints(const defaulttype::Vec<3, Real> & velocity,Real contactTime,const defaulttype::Vec<3,Real> * points,int n,defaulttype::Vec<3,Real> & proj_pt);

    static void projectPointOnCapsuleAndFindCapNormal(const defaulttype::Vec<3,Real> & pt,const defaulttype::Vec<3,Real> segment[2],Real radius,CapIntrConfiguration<Real> & capCfg,defaulttype::Vec<3,Real> & pt_on_capsule);

    static Real projectOnTriangle(defaulttype::Vec<3,Real> & pt,const defaulttype::Vec<3,Real> & t_p0,const defaulttype::Vec<3,Real> & t_p1,const defaulttype::Vec<3,Real> & t_p2,Real & s,Real & t);

    static defaulttype::Vec<3,Real> nearestPointOnSeg(const defaulttype::Vec<3,Real> & seg0,const defaulttype::Vec<3,Real> & seg1,const defaulttype::Vec<3,Real> & point);

    static void segNearestPoints(const defaulttype::Vec<3,Real> * p, const defaulttype::Vec<3,Real> * q,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q);

    static void segNearestPoints(const defaulttype::Vec<3,Real> & p0,const defaulttype::Vec<3,Real> & p1, const defaulttype::Vec<3,Real> & q0,const defaulttype::Vec<3,Real> & q1,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q);

    static void segNearestPoints(const defaulttype::Vec<3,Real> & p0,const defaulttype::Vec<3,Real> & p1, const defaulttype::Vec<3,Real> & q0,const defaulttype::Vec<3,Real> & q1,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q,
                                 SReal & alpha,SReal & beta);

    /**
      *Returns the squared distance between pt_on_face and pt_on_seg. Use only if the both faces lay on the same plane.
      */
    static Real facesNearestPoints(const defaulttype::Vec<3,Real> * first_face,int first_size,const defaulttype::Vec<3,Real> * second_face,int second_size,defaulttype::Vec<3,Real> & pt_on_first,defaulttype::Vec<3,Real> & pt_on_second);

    /**
      *Returns the squared distance between pt_on_face and pt_on_seg. Use only if the face and the segment lay on the same plane.
      */
    static Real faceSegNearestPoints(const defaulttype::Vec<3,Real> face[4],const defaulttype::Vec<3,Real> seg[2],defaulttype::Vec<3,Real> & pt_on_face,defaulttype::Vec<3,Real> & pt_on_seg);

    static Real faceSegNearestPoints(const defaulttype::Vec<3,Real> * face,int n,const defaulttype::Vec<3,Real> seg[2], defaulttype::Vec<3,Real> & pt_on_face,defaulttype::Vec<3,Real> & pt_on_seg);

    static bool equal(const defaulttype::Vec<3,Real> & vec0,const defaulttype::Vec<3,Real> & vec1);

    static bool nequal(Real a,Real b);

    static bool strInf(Real a,Real b);

    static bool inf(Real a,Real b);
};

template <class DataType>
struct IntrUtil<TOBB<DataType> >{
    typedef typename DataType::Real Real;
    typedef TOBB<DataType> Box;

    /**
      *Project pt on the Box box, caution,  works only if pt is outside the box.
      */
    static void project(defaulttype::Vec<3,Real> & pt,const Box & box);
};

//----------------------------------------------------------------------------
/**
  *IntrAxis is used to find the axis which maximizes the distance between the
  *two primitives, and, their configurations. Then the configurations are used to
  *find the contact points.
  */
template <class Primitive1Class,class Primitive2Class = Primitive1Class>
class IntrAxis;

/**
*The axis must be normalized when testing a capsule !.
*TDataTypes is the data type of the OBB.
*/
template <class TDataTypes>
class IntrAxis<TOBB<TDataTypes> >
{
public:
    typedef typename TDataTypes::Real Real;
    typedef TOBB<TDataTypes> Box;
    typedef typename TOBB<TDataTypes>::Coord Coord;
    typedef IntrConfiguration<Real> IntrConf;

    static bool Find (const Coord& axis,
        const Box& box0, const Box& box1,
        Real dmax,Real& dfirst,
        int& side, IntrConfiguration<Real>& box0CfgFinal,
        IntrConfiguration<Real>& box1CfgFinal,bool & config_modified);

    static bool Find (const Coord& axis,
        const defaulttype::Vec<3,Real> segment[2],Real radius, const Box& box,
        Real dmax, Real& dfirst,
        int& side, CapIntrConfiguration<Real> &segCfgFinal,
        IntrConfiguration<Real>& boxCfgFinal,bool & config_modified);
};

/**
  *IntrConfigManager is used to project the primitives on an axis and to find
  *the axis which maximizes the distance of the projected primitives. Each time you
  *run IntrConfigManager<Real>::Find with a new axis, config_modified is true if
  *the last passed axis maximizes the distance between the projection (described by IntrConfiguration) of the both primitives.
  */
template <typename TDataType>
struct IntrConfigManager;

template<class TDataTypes>
struct IntrConfigManager<TOBB<TDataTypes> >{
    typedef TOBB<TDataTypes> Box;
    typedef typename Box::Real Real;

    static void init(const defaulttype::Vec<3,Real> & axis,
                   const Box & box, IntrConfiguration<Real>& cfg);
};

template<typename Real>
struct IntrConfigManager{
    /**
    *The axis must be normalized when testing a capsule !.
    */
    static void init(const defaulttype::Vec<3,Real> & axis,
                    const defaulttype::Vec<3,Real> segment[2], Real radius,CapIntrConfiguration<Real>& cfg);

    static void init (const defaulttype::Vec<3,Real>& axis,
                    const defaulttype::Vec<3,Real> segment[2], IntrConfiguration<Real>& cfg);


    template <class Config0,class Config1>
    static bool Find (const Config0& cfg0Start,
        const Config1& cfg1Start,int& side,
        Config0& cfg0Final,
        Config1& cfg1Final, Real dmax,Real& dfirst,bool & config_modified);
};

//----------------------------------------------------------------------------
/**
  *Finds contact points between two primitives from their configuration and other parameters.
  */
template <class Primitive1Class,class Primitive2Class = Primitive1Class>
class FindContactSet;
/**
  *TDataTypes is the OBB type.
  */
template <class TDataTypes>
class  FindContactSet<TOBB<TDataTypes> >
{
public:
    typedef typename TDataTypes::Real Real;
    typedef TOBB<TDataTypes> Box;

    FindContactSet (const defaulttype::Vec<3,Real> segment[2], Real radius,const Box& box,const defaulttype::Vec<3,Real> & axis,
        int side, CapIntrConfiguration<Real> &capCfg,
        const IntrConfiguration<Real>& boxCfg,
        Real tfirst, defaulttype::Vec<3,Real> & pt_on_capsule,defaulttype::Vec<3,Real> & pt_on_box);

    FindContactSet (const Box& box0, const Box& box1,const defaulttype::Vec<3,Real> & axis,
        int side, const IntrConfiguration<Real>& box0Cfg,
        const IntrConfiguration<Real>& box1Cfg,
        Real tfirst,
        defaulttype::Vec<3,Real> & pt_on_first,defaulttype::Vec<3,Real> & pt_on_second);

private:

    /**
      *Function used by FindContactSet constructor when searching contact points between OBB and Capsule. segP0 is the apex of the
      *capsule segment which is the nearest to the OBB. This function is launched when the a semi-sphere is in intersection with the OBB.
      *The separating axis is axis, but it may be different after this method, it is stored in capCfg.
      */
    static void FindContactConfig(const defaulttype::Vec<3,Real> & axis,const defaulttype::Vec<3,Real> & segP0, Real radius,const Box & box,CapIntrConfiguration<Real> &capCfg,int side,
        defaulttype::Vec<3, Real> & pt_on_capsule,defaulttype::Vec<3, Real> & pt_on_box);
};
//----------------------------------------------------------------------------
// Miscellaneous support.
//----------------------------------------------------------------------------
// The input and output polygons are stored in P.  The size of P is
// assumed to be large enough to store the clipped convex polygon vertices.
// For now the maximum array size is 8 to support the current intersection
// algorithms.
template <typename Real>
void ClipConvexPolygonAgainstPlane (const defaulttype::Vec<3,Real>& normal,
    Real bonstant, int& quantity, defaulttype::Vec<3,Real>* P);

// Translates an index into the box back into real coordinates.
template <typename TReal>
defaulttype::Vec<3,TReal> GetPointFromIndex (int index, const MyBox<TReal>& box);

template <typename TDataTypes>
defaulttype::Vec<3,typename TDataTypes::Real> getPointFromIndex (int index, const TOBB<TDataTypes>& box);
//----------------------------------------------------------------------------

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_INTRUTILITY3_CPP)

extern template struct SOFA_BASE_COLLISION_API IntrUtil<SReal>;

#ifndef SOFA_FLOAT
extern template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<defaulttype::Rigid3dTypes> >;
extern template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<defaulttype::Rigid3dTypes> >;
extern template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<defaulttype::Rigid3dTypes> >;
extern template class SOFA_BASE_COLLISION_API IntrConfiguration<double>;
extern template struct SOFA_BASE_COLLISION_API IntrConfigManager<double>;
extern template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<defaulttype::Rigid3dTypes> >;
extern template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane(const defaulttype::Vec<3,SReal>&, SReal, int&,defaulttype::Vec<3,SReal>*);
extern template SOFA_BASE_COLLISION_API defaulttype::Vec<3,SReal> GetPointFromIndex (int, const MyBox<SReal>& );
extern template SOFA_BASE_COLLISION_API defaulttype::Vec<3,defaulttype::Rigid3dTypes::Real> getPointFromIndex (int, const TOBB<defaulttype::Rigid3dTypes>& );
extern template SOFA_BASE_COLLISION_API class CapIntrConfiguration<SReal>;
#endif
#ifndef SOFA_DOUBLE
extern template struct SOFA_BASE_COLLISION_API IntrUtil<float>;
extern template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<defaulttype::Rigid3fTypes> >;
extern template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<defaulttype::Rigid3fTypes> >;
extern template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<defaulttype::Rigid3fTypes> >;
extern template class SOFA_BASE_COLLISION_API IntrConfiguration<float>;
extern template struct SOFA_BASE_COLLISION_API IntrConfigManager<float>;
extern template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<defaulttype::Rigid3fTypes> >;
extern template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane(const defaulttype::Vec<3,float>&, float, int&,defaulttype::Vec<3,float>*);
extern template SOFA_BASE_COLLISION_API defaulttype::Vec<3,float> GetPointFromIndex (int, const MyBox<float>& );
extern template SOFA_BASE_COLLISION_API defaulttype::Vec<3,defaulttype::Rigid3fTypes::Real> getPointFromIndex (int, const TOBB<defaulttype::Rigid3fTypes>& );
extern template SOFA_BASE_COLLISION_API class CapIntrConfiguration<float>;
#endif
#endif

}
}
}

#ifndef SOFA_EXTERN_TEMPLATE
#include <SofaBaseCollision/IntrUtility3.inl>
#endif

#endif
