/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMeshCollision/IntrMeshUtility.h>
#include <SofaBaseCollision/IntrUtility3.inl>


namespace sofa{
namespace component{
namespace collision{


template <class DataTypes>
void IntrUtil<TTriangle<DataTypes> >::triFaceNearestPoints(const IntrTri & tri,const defaulttype::Vec<3,Real> *second_face,int second_size, defaulttype::Vec<3,Real> &pt_on_first, defaulttype::Vec<3,Real> &pt_on_second){
    Real min1 = std::numeric_limits<Real>::max();
    Real min2 = std::numeric_limits<Real>::max();
    Real new_min;
    int first_index1 = -1;
    int first_index2 = -1;
    int second_index1 = -1;
    int second_index2 = -1;

    for(int i = 0 ; i < 3 ; ++i){
        for(int j = 0 ; j < second_size ; ++j){
            new_min = (tri[i] - second_face[j]).norm2();
            if(min1 > new_min){
                min2 = min1;
                min1 = new_min;
                first_index2 = first_index1;
                second_index2 = second_index1;

                first_index1 = i;
                second_index1 = j;
            }
            else if(min2 > new_min){
                first_index2 = i;
                second_index2 = j;
                min2 = new_min;
            }
        }
    }

    if(min2 > min1 - IntrUtil<Real>::ZERO_TOLERANCE() && min2 < min1 + IntrUtil<Real>::ZERO_TOLERANCE()){
        pt_on_first = (tri[first_index1] + tri[first_index2])/2.0;
        pt_on_second = (second_face[second_index1] + second_face[second_index2])/2.0;
    }
    else{
        pt_on_first = tri[first_index1];
        pt_on_second = second_face[second_index1];
    }
}

template <typename DataType>
SReal IntrUtil<TTriangle<DataType> >::triSegNearestPoints(const IntrTri & tri,const defaulttype::Vec<3,Real> seg[2], defaulttype::Vec<3,Real> & pt_on_face,defaulttype::Vec<3,Real> & pt_on_seg){
    Real min = std::numeric_limits<Real>::max();
    defaulttype::Vec<3,Real> cur_pt_on_face,cur_pt_on_seg;
    defaulttype::Vec<3,Real> face_seg[2];
    Real new_min;

    for(int j = 0 ; j < 3 ; ++j){
        face_seg[0] = tri.p(j);
        if(j < 2){
            face_seg[1] = tri.p(j + 1);
        }
        else{
            face_seg[1] = tri.p(0);
        }

        IntrUtil<Real>::segNearestPoints(face_seg,seg,cur_pt_on_face,cur_pt_on_seg);

        if((new_min = (cur_pt_on_face - cur_pt_on_seg).norm2()) < min){
            min = new_min;
            pt_on_face = cur_pt_on_face;
            pt_on_seg = cur_pt_on_seg;
        }
    }

    return min;
}


template <class DataType>
typename IntrUtil<TTriangle<DataType> >::Real IntrUtil<TTriangle<DataType> >::project(defaulttype::Vec<3,Real> & pt,const TTriangle<DataType> & tri){
    Real s,t;
    return IntrUtil<Real>::projectOnTriangle(pt,tri.p1(),tri.p2(),tri.p3(),s,t);
}


template <class TDataTypes1,class TDataTypes2>
bool IntrAxis<TTriangle<TDataTypes1>,TOBB<TDataTypes2> >::Find (const Coord& axis,
    const IntrTri & triangle, const Box& box,
    Real dmax, Real& tfirst,
    int& side, IntrConfiguration<Real>& triCfgFinal,
    IntrConfiguration<Real>& boxCfgFinal,bool & config_modified)
{
    IntrConfiguration<Real> triCfgStart;
    IntrConfigManager<IntrTri>::init(axis, triangle, triCfgStart);

    IntrConfiguration<Real> boxCfgStart;
    IntrConfigManager<Box>::init(axis, box, boxCfgStart);

    return IntrConfigManager<Real>::Find(triCfgStart, boxCfgStart, side,
        triCfgFinal, boxCfgFinal, dmax,tfirst, config_modified);
}


//template <class DataTypes>
//void IntrConfigManager<TTriangle<DataTypes> >::init (const Coord& axis,
//    const IntrTri & triangle, IntrConfiguration<Real>& cfg)
//{
//    // Find projections of vertices onto potential separating axis.
//    Real d0 = axis * triangle.p1();
//    Real d1 = axis * triangle.p2();
//    Real d2 = axis * triangle.p3();

//    // Explicit sort of vertices to construct a IntrConfiguration.
//    if (d0 <= d1)
//    {
//        if (d1 <= d2) // D0 <= D1 <= D2
//        {
//            if (d0 != d1)
//            {
//                if (d1 != d2)
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m111;
//                }
//                else
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m12;
//                }
//            }
//            else // ( D0 == D1 )
//            {
//                if (d1 != d2)
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m21;
//                }
//                else
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m3;
//                }
//            }
//            cfg.mIndex[0] = 0;
//            cfg.mIndex[1] = 1;
//            cfg.mIndex[2] = 2;
//            cfg.mMin = d0;
//            cfg.mMax = d2;
//        }
//        else if (d0 <= d2) // D0 <= D2 < D1
//        {
//            if (d0 != d2)
//            {
//                cfg.mMap = IntrConfiguration<Real>::m111;
//                cfg.mIndex[0] = 0;
//                cfg.mIndex[1] = 2;
//                cfg.mIndex[2] = 1;
//            }
//            else
//            {
//                cfg.mMap = IntrConfiguration<Real>::m21;
//                cfg.mIndex[0] = 2;
//                cfg.mIndex[1] = 0;
//                cfg.mIndex[2] = 1;
//            }
//            cfg.mMin = d0;
//            cfg.mMax = d1;
//        }
//        else // D2 < D0 <= D1
//        {
//            if (d0 != d1)
//            {
//                cfg.mMap = IntrConfiguration<Real>::m111;
//            }
//            else
//            {
//                cfg.mMap = IntrConfiguration<Real>::m12;
//            }

//            cfg.mIndex[0] = 2;
//            cfg.mIndex[1] = 0;
//            cfg.mIndex[2] = 1;
//            cfg.mMin = d2;
//            cfg.mMax = d1;
//        }
//    }
//    else if (d2 <= d1) // D2 <= D1 < D0
//    {
//        if (d2 != d1)
//        {
//            cfg.mMap = IntrConfiguration<Real>::m111;
//            cfg.mIndex[0] = 2;
//            cfg.mIndex[1] = 1;
//            cfg.mIndex[2] = 0;
//        }
//        else
//        {
//            cfg.mMap = IntrConfiguration<Real>::m21;
//            cfg.mIndex[0] = 1;
//            cfg.mIndex[1] = 2;
//            cfg.mIndex[2] = 0;

//        }
//        cfg.mMin = d2;
//        cfg.mMax = d0;
//    }
//    else if (d2 <= d0) // D1 < D2 <= D0
//    {
//        if (d2 != d0)
//        {
//            cfg.mMap = IntrConfiguration<Real>::m111;
//        }
//        else
//        {
//            cfg.mMap = IntrConfiguration<Real>::m12;
//        }

//        cfg.mIndex[0] = 1;
//        cfg.mIndex[1] = 2;
//        cfg.mIndex[2] = 0;
//        cfg.mMin = d1;
//        cfg.mMax = d0;
//    }
//    else // D1 < D0 < D2
//    {
//        cfg.mMap = IntrConfiguration<Real>::m111;
//        cfg.mIndex[0] = 1;
//        cfg.mIndex[1] = 0;
//        cfg.mIndex[2] = 2;
//        cfg.mMin = d1;
//        cfg.mMax = d2;
//    }
//}


template <class DataTypes>
void IntrConfigManager<TTriangle<DataTypes> >::init (const Coord& axis,
    const IntrTri & triangle, IntrConfiguration<Real>& cfg)
{
    // Find projections of vertices onto potential separating axis.
    Real d0 = axis * triangle.p1();
    Real d1 = axis * triangle.p2();
    Real d2 = axis * triangle.p3();

    // Explicit sort of vertices to construct a IntrConfiguration.
    if (IntrUtil<Real>::inf(d0,d1))
    {
        if (IntrUtil<Real>::inf(d1,d2)) // D0 <= D1 <= D2
        {
            if (IntrUtil<Real>::nequal(d0,d1))
            {
                if (IntrUtil<Real>::nequal(d1,d2))
                {
                    cfg.mMap = IntrConfiguration<Real>::m111;
                }
                else
                {
                    cfg.mMap = IntrConfiguration<Real>::m12;
                }
            }
            else // ( D0 == D1 )
            {
                if (IntrUtil<Real>::nequal(d1,d2))
                {
                    cfg.mMap = IntrConfiguration<Real>::m21;
                }
                else
                {
                    cfg.mMap = IntrConfiguration<Real>::m3;
                }
            }
            cfg.mIndex[0] = 0;
            cfg.mIndex[1] = 1;
            cfg.mIndex[2] = 2;
            cfg.mMin = d0;
            cfg.mMax = d2;
        }
        else if (IntrUtil<Real>::inf(d0,d2)) // D0 <= D2 < D1
        {
            if (IntrUtil<Real>::nequal(d0,d2))
            {
                cfg.mMap = IntrConfiguration<Real>::m111;
                cfg.mIndex[0] = 0;
                cfg.mIndex[1] = 2;
                cfg.mIndex[2] = 1;
            }
            else
            {
                cfg.mMap = IntrConfiguration<Real>::m21;
                cfg.mIndex[0] = 2;
                cfg.mIndex[1] = 0;
                cfg.mIndex[2] = 1;
            }
            cfg.mMin = d0;
            cfg.mMax = d1;
        }
        else // D2 < D0 <= D1
        {
            if (IntrUtil<Real>::nequal(d0,d1))
            {
                cfg.mMap = IntrConfiguration<Real>::m111;
            }
            else
            {
                cfg.mMap = IntrConfiguration<Real>::m12;
            }

            cfg.mIndex[0] = 2;
            cfg.mIndex[1] = 0;
            cfg.mIndex[2] = 1;
            cfg.mMin = d2;
            cfg.mMax = d1;
        }
    }
    else if (IntrUtil<Real>::inf(d2,d1)) // D2 <= D1 < D0
    {
        if (IntrUtil<Real>::nequal(d2,d1))
        {
            cfg.mMap = IntrConfiguration<Real>::m111;
            cfg.mIndex[0] = 2;
            cfg.mIndex[1] = 1;
            cfg.mIndex[2] = 0;
        }
        else
        {
            cfg.mMap = IntrConfiguration<Real>::m21;
            cfg.mIndex[0] = 1;
            cfg.mIndex[1] = 2;
            cfg.mIndex[2] = 0;

        }
        cfg.mMin = d2;
        cfg.mMax = d0;
    }
    else if (IntrUtil<Real>::inf(d2,d0)) // D1 < D2 <= D0
    {
        if (IntrUtil<Real>::nequal(d2,d0))
        {
            cfg.mMap = IntrConfiguration<Real>::m111;
        }
        else
        {
            cfg.mMap = IntrConfiguration<Real>::m12;
        }

        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 2;
        cfg.mIndex[2] = 0;
        cfg.mMin = d1;
        cfg.mMax = d0;
    }
    else // D1 < D0 < D2
    {
        cfg.mMap = IntrConfiguration<Real>::m111;
        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 0;
        cfg.mIndex[2] = 2;
        cfg.mMin = d1;
        cfg.mMax = d2;
    }
}

template <class TDataTypes1,class TDataTypes2>
FindContactSet<TTriangle<TDataTypes1>,TOBB<TDataTypes2> >::FindContactSet (const IntrTri& tri,
    const Box& box, const defaulttype::Vec<3,Real> & axis,int side, const IntrConfiguration<Real>& triCfg,
    const IntrConfiguration<Real>& boxCfg,Real tfirst,
    defaulttype::Vec<3,Real> & pt_on_tri,defaulttype::Vec<3,Real> & pt_on_box)
{
    const int* tIndex = triCfg.mIndex;
    const int* bIndex = boxCfg.mIndex;

    if (side == IntrConfiguration<Real>::LEFT)
    {
        // box on left of tri
        if (triCfg.mMap == IntrConfiguration<Real>::m111
        ||  triCfg.mMap == IntrConfiguration<Real>::m12)//triangle's vertex
        {
            //P[0] = triFinal[tIndex[0]];
            pt_on_tri = tri.p(tIndex[0]);
            pt_on_box = pt_on_tri;
            IntrUtil<Box>::project(pt_on_box,box);
            //assert(box.onSurface(pt_on_box));
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)//box's vertex
        {
            //P[0] = GetPointFromIndex(bIndex[7], boxFinal);
            pt_on_box = getPointFromIndex(bIndex[7], box);
            pt_on_tri = pt_on_box;
            IntrUtil<IntrTri>::project(pt_on_tri,tri);
            //assert(box.onSurface(pt_on_box));
        }
        else if (triCfg.mMap == IntrConfiguration<Real>::m21)
        {
            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // triseg-boxseg intersection
                defaulttype::Vec<3,Real> triSeg[2], boxSeg[2];
                triSeg[0] = tri.p(tIndex[0]);
                triSeg[1] = tri.p(tIndex[1]);
                boxSeg[0] = getPointFromIndex(bIndex[6], box);
                boxSeg[1] = getPointFromIndex(bIndex[7], box);
                IntrUtil<Real>::segNearestPoints(triSeg,boxSeg,pt_on_tri,pt_on_box);                
                //assert(box.onSurface(pt_on_box));
            }
            else // boxCfg.mMap == IntrConfiguration<Real>::m44, triangles'edge box's face
            {
                int quantity;
                defaulttype::Vec<3,Real> P[2];

                defaulttype::Vec<3,Real> triFinal[3] =
                {
                    tri.p(0) - tfirst*axis,
                    tri.p(1) - tfirst*axis,
                    tri.p(2) - tfirst*axis,
                };

                // triseg-boxface intersection
                defaulttype::Vec<3,Real> triSeg[2];
                defaulttype::Vec<3,Real> boxFace[4];
                triSeg[0] = triFinal[tIndex[0]];
                triSeg[1] = triFinal[tIndex[1]];
                boxFace[0] = getPointFromIndex(bIndex[4], box);
                boxFace[1] = getPointFromIndex(bIndex[5], box);
                boxFace[2] = getPointFromIndex(bIndex[6], box);
                boxFace[3] = getPointFromIndex(bIndex[7], box);
                IntrUtil<Real>::CoplanarSegmentRectangle(triSeg, boxFace, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri - axis * tfirst;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
//                    triSeg[0] = tri.p(tIndex[0]);
//                    triSeg[1] = tri.p(tIndex[1]);
//                    IntrUtil<Real>::faceSegNearestPoints(boxFace,triSeg,pt_on_box,pt_on_tri);
                    IntrUtil<Real>::faceSegNearestPoints(boxFace,triSeg,pt_on_box,pt_on_tri);
                    pt_on_tri += tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
            }
        }
        else // triCfg.mMap == IntrConfiguration<Real>::m3, triangle's face
        {
            int quantity;
            defaulttype::Vec<3,Real> P[6];

            defaulttype::Vec<3,Real> triFinal[3] =
            {
                tri.p(0) - tfirst*axis,
                tri.p(1) - tfirst*axis,
                tri.p(2) - tfirst*axis,
            };

            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)//box's edge
            {
                // boxseg-triface intersection
                defaulttype::Vec<3,Real> boxSeg[2];
                boxSeg[0] = getPointFromIndex(bIndex[6], box);
                boxSeg[1] = getPointFromIndex(bIndex[7], box);
                IntrUtil<Real>::ColinearSegmentTriangle(boxSeg, triFinal, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri - tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
                    IntrUtil<IntrTri>::triSegNearestPoints(tri,boxSeg,pt_on_tri,pt_on_box);
                    //assert(box.onSurface(pt_on_box));
                }
            }
            else
            {
                // triface-boxface intersection
                defaulttype::Vec<3,Real> boxFace[4];
                boxFace[0] = getPointFromIndex(bIndex[4], box);
                boxFace[1] = getPointFromIndex(bIndex[5], box);
                boxFace[2] = getPointFromIndex(bIndex[6], box);
                boxFace[3] = getPointFromIndex(bIndex[7], box);

                IntrUtil<Real>::CoplanarTriangleRectangle(triFinal, boxFace, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri - tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
                    IntrUtil<IntrTri>::triFaceNearestPoints(tri,boxFace,4,pt_on_tri,pt_on_box);
                    //assert(box.onSurface(pt_on_box));
                }
            }
        }
    }
    else // side == RIGHT
    {
        // box on right of tri
        if (triCfg.mMap == IntrConfiguration<Real>::m111
        ||  triCfg.mMap == IntrConfiguration<Real>::m21)//triangle's vertex
        {
            pt_on_tri = tri.p(tIndex[2]);
            pt_on_box = pt_on_tri;
            IntrUtil<Box>::project(pt_on_box,box);
            //assert(box.onSurface(pt_on_box));
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)//box's vertex
        {
            pt_on_box = getPointFromIndex(bIndex[0], box);
            pt_on_tri = pt_on_box;
            IntrUtil<IntrTri>::project(pt_on_tri,tri);
            //assert(box.onSurface(pt_on_box));
        }
        else if (triCfg.mMap == IntrConfiguration<Real>::m12)//triangle's edge
        {
            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)//box's edge
            {
                // segment-segment intersection
                defaulttype::Vec<3,Real> triSeg[2], boxSeg[2];
                triSeg[0] = tri[tIndex[1]];
                triSeg[1] = tri[tIndex[2]];
                boxSeg[0] = getPointFromIndex(bIndex[0], box);
                boxSeg[1] = getPointFromIndex(bIndex[1], box);

                IntrUtil<Real>::segNearestPoints(triSeg,boxSeg,pt_on_tri,pt_on_box);
                //assert(box.onSurface(pt_on_box));
            }
            else // boxCfg.mMap == IntrConfiguration<Real>::m44, box's face
            {
                defaulttype::Vec<3,Real> P[2];
                int quantity;
                defaulttype::Vec<3,Real> triFinal[3] =
                {
                    tri.p(0) + tfirst*axis,
                    tri.p(1) + tfirst*axis,
                    tri.p(2) + tfirst*axis,
                };

                // triseg-boxface intersection
                defaulttype::Vec<3,Real> triSeg[2], boxFace[4];
                triSeg[0] = triFinal[tIndex[1]];
                triSeg[1] = triFinal[tIndex[2]];

                boxFace[0] = getPointFromIndex(bIndex[0], box);
                boxFace[1] = getPointFromIndex(bIndex[1], box);
                boxFace[2] = getPointFromIndex(bIndex[2], box);
                boxFace[3] = getPointFromIndex(bIndex[3], box);


                IntrUtil<Real>::CoplanarSegmentRectangle(triSeg, boxFace, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri + tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
//                    triSeg[0] = tri[tIndex[1]];
//                    triSeg[1] = tri[tIndex[2]];
//                    IntrUtil<Real>::faceSegNearestPoints(boxFace,4,triSeg,pt_on_box,pt_on_tri);
                    IntrUtil<Real>::faceSegNearestPoints(boxFace,4,triSeg,pt_on_box,pt_on_tri);
                    pt_on_tri -= tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
            }
        }
        else // triCfg.mMap == IntrConfiguration<Real>::m3
        {
            defaulttype::Vec<3,Real> P[6];
            int quantity;
            defaulttype::Vec<3,Real> triFinal[3] =
            {
                tri.p(0) + tfirst*axis,
                tri.p(1) + tfirst*axis,
                tri.p(2) + tfirst*axis,
            };

            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // boxseg-triface intersection
                defaulttype::Vec<3,Real> boxSeg[2];
                boxSeg[0] = getPointFromIndex(bIndex[0], box);
                boxSeg[1] = getPointFromIndex(bIndex[1], box);

                IntrUtil<Real>::ColinearSegmentTriangle(boxSeg, triFinal, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri + tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
                    IntrUtil<IntrTri>::triSegNearestPoints(tri,boxSeg,pt_on_tri,pt_on_box);
                    //assert(box.onSurface(pt_on_box));
                }
            }
            else
            {
                // triface-boxface intersection
                defaulttype::Vec<3,Real> boxFace[4];
                boxFace[0] = getPointFromIndex(bIndex[0], box);
                boxFace[1] = getPointFromIndex(bIndex[1], box);
                boxFace[2] = getPointFromIndex(bIndex[2], box);
                boxFace[3] = getPointFromIndex(bIndex[3], box);

                IntrUtil<Real>::CoplanarTriangleRectangle(triFinal, boxFace, quantity, P);

                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_tri);
                    pt_on_box = pt_on_tri + tfirst * axis;
                    //assert(box.onSurface(pt_on_box));
                }
                else{
                    IntrUtil<IntrTri>::triFaceNearestPoints(tri,boxFace,4,pt_on_tri,pt_on_box);
                    //assert(box.onSurface(pt_on_box));
                }
            }
        }
    }
}


}
}
}
