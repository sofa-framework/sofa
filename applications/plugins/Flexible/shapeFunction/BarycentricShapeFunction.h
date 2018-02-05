/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef FLEXIBLE_BarycentricShapeFunction_H
#define FLEXIBLE_BarycentricShapeFunction_H

#include <Flexible/config.h>
#include "../shapeFunction/BaseShapeFunction.h"
#include <sofa/core/topology/BaseMeshTopology.h>

#include <algorithm>
#include <iostream>

namespace sofa
{
namespace component
{
namespace shapefunction
{

/**
Barycentric shape functions are the barycentric coordinates of points inside cells (can be edges, triangles, quads, tetrahedra, hexahedra)
  */

template <class ShapeFunctionTypes_>
class BarycentricShapeFunction : public core::behavior::BaseShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BarycentricShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(core::behavior::BaseShapeFunction, ShapeFunctionTypes_));
    typedef core::behavior::BaseShapeFunction<ShapeFunctionTypes_> Inherit;

    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VCoord VCoord;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::Cell Cell;

    typedef sofa::core::topology::BaseMeshTopology Topo;
    SingleLink<BarycentricShapeFunction<ShapeFunctionTypes_>, Topo, 0> parentTopology;

    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::Hessian Hessian;
    enum {spatial_dimensions=Inherit::spatial_dimensions};
    typedef defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> Basis;
    sofa::helper::vector<Basis> bases;
    Data< Real > f_tolerance;
    Cell cellIndex;  ///< used by external classes to retrieve the index of the cell where barycentric weights are computed from


    void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)
    {
        InternalShapeFunction<spatial_dimensions>::computeShapeFunction( this, childPosition, this->cellIndex, ref, w, dw, ddw, cell );
    }

protected:
    // 3D space
    template<int spatial_dimensions, class DUMMY=void/*hack since template specialization of a nested class is not possible without specialization of the base class, but partial specialization is... */>
    struct InternalShapeFunction
    {
        static void getBasisFrom2DElements( Basis& b, const Coord& p0, const Coord& p1, const Coord& p2 )
        {
            b[0] = p1 - p0;
            b[1] = p2 - p0;
            b[2] = cross ( b[0], b[1] );
        }


        static double getDistanceTriangle( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-(Real)0.01,v[0]+v[1]-(Real)1. ) );
        }
        static double getDistanceQuad( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-(Real)1.,v[0]-(Real)1. ),std::max ( v[2]-(Real)0.01,-v[2]-(Real)0.01 ) ) );
        }
        static double getDistanceTetra( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-(Real)1. ) );
        }
        static double getDistanceHexa( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-(Real)1. ),std::max ( v[1]-1,v[2]-(Real)1. ) ) );
        }


        /// computes w such that x= p0*(1-wx)*(1-wz)*(1-wy) + p1*wx*(1-wz)*(1-wy) + p5*wx*wz*(1-wy) + p4*(1-wx)*wz*(1-wy) + p3*(1-wx)*(1-wz)*wy + p2*wx*(1-wz)*wy + p6*wx*wz*wy + p7*(1-wx)*wz*wy
        /// using Newton method
        static void computeHexaTrilinearWeights(Coord &w, const Coord p[8], const Coord &x,const Real &tolerance)
        {
            w[0]=0.5; w[1]=0.5; w[2]=0.5; // initial guess
            static const unsigned int MAXIT=20;
            static const Real MIN_DETERMINANT = 1.0e-100;
            unsigned int it=0;
            while( it < MAXIT)
            {
                Coord g(1.-w[0],1.-w[1],1.-w[2]);
                Coord f = p[0]*g[0]*g[2]*g[1] + p[1]*w[0]*g[2]*g[1] + p[5]*w[0]*w[2]*g[1] + p[4]*g[0]*w[2]*g[1] + p[3]*g[0]*g[2]*w[1] + p[2]*w[0]*g[2]*w[1] + p[6]*w[0]*w[2]*w[1] + p[7]*g[0]*w[2]*w[1] - x; // function to minimize
                if(f.norm2()<tolerance) {  return; }

                Basis df;
                df[0] = - p[0]*g[2]*g[1] + p[1]*g[2]*g[1] + p[5]*w[2]*g[1] - p[4]*w[2]*g[1] - p[3]*g[2]*w[1] + p[2]*g[2]*w[1] + p[6]*w[2]*w[1] - p[7]*w[2]*w[1];
                df[1] = - p[0]*g[0]*g[2] - p[1]*w[0]*g[2] - p[5]*w[0]*w[2] - p[4]*g[0]*w[2] + p[3]*g[0]*g[2] + p[2]*w[0]*g[2] + p[6]*w[0]*w[2] + p[7]*g[0]*w[2];
                df[2] = - p[0]*g[0]*g[1] - p[1]*w[0]*g[1] + p[5]*w[0]*g[1] + p[4]*g[0]*g[1] - p[3]*g[0]*w[1] - p[2]*w[0]*w[1] + p[6]*w[0]*w[1] + p[7]*g[0]*w[1];

                Real det=determinant(df);
                if ( -MIN_DETERMINANT<=det && det<=MIN_DETERMINANT) { return; }
                Basis dfinv;
                dfinv(0,0)= (df(1,1)*df(2,2) - df(2,1)*df(1,2))/det;
                dfinv(0,1)= (df(1,2)*df(2,0) - df(2,2)*df(1,0))/det;
                dfinv(0,2)= (df(1,0)*df(2,1) - df(2,0)*df(1,1))/det;
                dfinv(1,0)= (df(2,1)*df(0,2) - df(0,1)*df(2,2))/det;
                dfinv(1,1)= (df(2,2)*df(0,0) - df(0,2)*df(2,0))/det;
                dfinv(1,2)= (df(2,0)*df(0,1) - df(0,0)*df(2,1))/det;
                dfinv(2,0)= (df(0,1)*df(1,2) - df(1,1)*df(0,2))/det;
                dfinv(2,1)= (df(0,2)*df(1,0) - df(1,2)*df(0,0))/det;
                dfinv(2,2)= (df(0,0)*df(1,1) - df(1,0)*df(0,1))/det;

                w -= dfinv*f;
                it++;
            }
        }


        static void init( BarycentricShapeFunction<ShapeFunctionTypes_>* B )
        {
            helper::ReadAccessor<Data<helper::vector<Coord> > > parent(B->f_position);
            if(!parent.size()) { B->serr<<"Parent nodes not found"<<B->sendl; return; }

            const Topo::SeqTetrahedra& tetrahedra = B->parentTopology->getTetrahedra();
            const Topo::SeqHexahedra& cubes = B->parentTopology->getHexahedra();
            const Topo::SeqTriangles& triangles = B->parentTopology->getTriangles();
            const Topo::SeqQuads& quads = B->parentTopology->getQuads();
            const Topo::SeqEdges& edges = B->parentTopology->getEdges();

            if ( tetrahedra.empty() && cubes.empty() )
            {
                if ( triangles.empty() && quads.empty() )
                {
                    if ( edges.empty() ) return;

                    //no 3D elements, nor 2D elements -> map on 1D elements
                    B->f_nbRef.setValue(2);
                    B->bases.resize ( edges.size() );
                    for (unsigned int e=0; e<edges.size(); e++ )
                    {
                        Coord V12 = ( parent[edges[e][1]]-parent[edges[e][0]] );
                        B->bases[e][0] = V12/V12.norm2();
                    }
                }
                else
                {
                    // no 3D elements -> map on 2D elements
                    if(quads.size()) B->f_nbRef.setValue(4);
                    else B->f_nbRef.setValue(3);
                    B->bases.resize ( triangles.size() +quads.size() );
                    for ( unsigned int t = 0; t < triangles.size(); t++ )
                    {
                        Basis m,mt;
                        getBasisFrom2DElements( m, parent[triangles[t][0]], parent[triangles[t][1]], parent[triangles[t][2]] );
                        mt.transpose ( m );
                        B->bases[t].invert ( mt );
                    }
                    int c0 = triangles.size();
                    for ( unsigned int c = 0; c < quads.size(); c++ )
                    {
                        Basis m,mt;
                        getBasisFrom2DElements( m, parent[quads[c][0]], parent[quads[c][1]], parent[quads[c][3]] );
                        mt.transpose ( m );
                        B->bases[c0+c].invert ( mt );
                    }
                }
            }
            else
            {
                // map on 3D elements
                if(cubes.size()) B->f_nbRef.setValue(8);
                else B->f_nbRef.setValue(4);
                B->bases.resize ( tetrahedra.size() +cubes.size() );
                for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
                {
                    Basis m,mt;
                    m[0] = parent[tetrahedra[t][1]]-parent[tetrahedra[t][0]];
                    m[1] = parent[tetrahedra[t][2]]-parent[tetrahedra[t][0]];
                    m[2] = parent[tetrahedra[t][3]]-parent[tetrahedra[t][0]];
                    mt.transpose ( m );
                    B->bases[t].invert ( mt );
                }
                int c0 = tetrahedra.size();
                for ( unsigned int c = 0; c < cubes.size(); c++ )
                {
                    Basis m,mt;
                    m[0] = parent[cubes[c][1]]-parent[cubes[c][0]];
                    m[1] = parent[cubes[c][3]]-parent[cubes[c][0]];
                    m[2] = parent[cubes[c][4]]-parent[cubes[c][0]];
                    mt.transpose ( m );
                    B->bases[c0+c].invert ( mt );
                }
            }
        }

        static void computeShapeFunction( const BarycentricShapeFunction<ShapeFunctionTypes_>* B, const Coord& childPosition, Cell &index, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)
        {
            // resize input
            unsigned int nbRef=B->f_nbRef.getValue();
            ref.resize(nbRef); ref.fill(0);
            w.resize(nbRef); w.fill(0);
            if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
            if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].clear(); }

            // get parent topology and nodes
            if(!B->parentTopology) return;

            helper::ReadAccessor<Data<helper::vector<Coord> > > parent(B->f_position);
            if(!parent.size()) return;

            const Topo::SeqTetrahedra& tetrahedra = B->parentTopology->getTetrahedra();
            const Topo::SeqHexahedra& cubes = B->parentTopology->getHexahedra();
            const Topo::SeqTriangles& triangles = B->parentTopology->getTriangles();
            const Topo::SeqQuads& quads = B->parentTopology->getQuads();
            const Topo::SeqEdges& edges = B->parentTopology->getEdges();

            // compute barycentric weights by projection in cell basis
            index = -1;
            double distance = -B->f_tolerance.getValue();
            Coord coefs;

            if ( tetrahedra.empty() && cubes.empty() )
            {
                if ( triangles.empty() && quads.empty() )
                {
                    if ( edges.empty() ) return;
                    //no 3D elements, nor 2D elements -> map on 1D elements
                    for ( unsigned int i = 0; i < edges.size(); i++ )
                        if(cell==-1 || cell==(int)i)
                        {
                            Coord v = B->bases[i] * ( childPosition - parent[edges[i][0]] );
                            double d = std::max ( -v[0], v[0]-(Real)1. );
                            if ( d<=distance ) { coefs = v; distance = d; index = i; }
                        }
                    if ( index!=-1 ) // addPointInLine
                    {
                        ref[0]=edges[index][0]; ref[1]=edges[index][1];
                        // weights
                        w[0]=(Real)1.-coefs[0]; w[1]=coefs[0];
                        if(dw) {  (*dw)[0]=-B->bases[index][0]; (*dw)[1]=B->bases[index][0]; }
                    }
                }
                else
                {
                    // no 3D elements -> map on 2D elements
                    for ( unsigned int i = 0; i < triangles.size(); i++ )
                        if(cell==-1 || cell==(int)i)
                        {
                            Coord v = B->bases[i] * ( childPosition - parent[triangles[i][0]] );
                            double d = getDistanceTriangle( v );
                            if ( d<=distance ) { coefs = v; distance = d; index = i; }
                        }
                    int c0 = triangles.size();
                    for ( unsigned int i = 0; i < quads.size(); i++ )
                        if(cell==-1 || cell==(int)(i+c0))
                        {
                            Coord v = B->bases[c0+i] * ( childPosition - parent[quads[i][0]] );
                            double d = getDistanceQuad( v );
                            if ( d<=distance ) { coefs = v; distance = d; index = c0+i; }
                        }
                    if ( index!=-1 && index<c0 ) // addPointInTriangle
                    {
                        ref[0]=triangles[index][0];                          ref[1]=triangles[index][1];  ref[2]=triangles[index][2];
                        // weights
                        w[0]=(Real)1.-coefs[0]-coefs[1];                     w[1]=coefs[0];               w[2]=coefs[1];
                        if(dw) {  (*dw)[0]=-B->bases[index][0]-B->bases[index][1]; (*dw)[1]=B->bases[index][0];    (*dw)[2]=B->bases[index][1]; }
                    }
                    else if ( index!=-1 ) // addPointInQuad
                    {
                        Real fx=coefs[0],fy=coefs[1];
                        Real gx=(Real)1.-fx,gy=(Real)1.-fy;
                        Gradient dfx=B->bases[index-c0][0],dfy=B->bases[index-c0][1];
                        Gradient dgx=-dfx,dgy=-dfy;
                        for ( unsigned int i = 0; i < 4; i++ ) ref[i]=quads[index-c0][i];
                        // weights
                        w[0]=gx*gy; w[1]=fx*gy; w[2]=fx*fy; w[3]=gx*fy;
                        if(dw)
                        {
                            (*dw)[0]=dgx*gy+dgy*gx;
                            (*dw)[1]=dfx*gy+dgy*fx;
                            (*dw)[2]=dfx*fy+dfy*fx;
                            (*dw)[3]=dgx*fy+dfy*gx;
                        }
                        if(ddw)
                        {
                            (*ddw)[0]=B->covs(dgx,dgy);
                            (*ddw)[1]=B->covs(dfx,dgy);
                            (*ddw)[2]=B->covs(dfx,dfy);
                            (*ddw)[3]=B->covs(dgx,dfy);
                        }
                    }
                }
            }
            else
            {
                // map on 3D elements
                for ( unsigned int i = 0; i < tetrahedra.size(); i++ )
                    if(cell==-1 || cell==(int)i)
                    {
                        Coord v = B->bases[i] * ( childPosition - parent[tetrahedra[i][0]] );
                        double d = getDistanceTetra( v );
                        if ( d<=distance ) { coefs = v; distance = d; index = i; }
                    }
                int c0 = tetrahedra.size();
                for ( unsigned int i = 0; i < cubes.size(); i++ )
                    if(cell==-1 || cell==(int)(i+c0))
                    {
                        Coord v = B->bases[c0+i] * ( childPosition - parent[cubes[i][0]] );  // for cuboid hexahedra
//                        Coord v; Coord ph[8];  for ( unsigned int j = 0; j < 8; j++ ) ph[j]=parent[cubes[i][j]]; computeHexaTrilinearWeights(v,ph,childPosition,1E-10); // for arbitrary hexahedra
                        double d = getDistanceHexa( v );
                        if ( d<=distance ) { coefs = v; distance = d; index = c0+i; }
                    }
                if ( index!=-1 && index<c0 ) // addPointInTet
                {
                    ref[0]=tetrahedra[index][0];                                         ref[1]=tetrahedra[index][1];  ref[2]=tetrahedra[index][2];    ref[3]=tetrahedra[index][3];
                    // weights
                    w[0]=(Real)1.-coefs[0]-coefs[1]-coefs[2];                            w[1]=coefs[0];                w[2]=coefs[1];                  w[3]=coefs[2];
                    if(dw) {
                        Gradient b0=B->bases[index][0],b1=B->bases[index][1],b2=B->bases[index][2];
                        (*dw)[0]=-b0-b1-b2; (*dw)[1]=b0;     (*dw)[2]=b1;       (*dw)[3]=b2;
                    }
                }
                else if ( index!=-1 ) // addPointInHex
                {
                    Real fx=coefs[0],fy=coefs[1],fz=coefs[2];
                    Real gx=(Real)1.-fx,gy=(Real)1.-fy,gz=(Real)1.-fz;
                    Coord dfx=B->bases[index-c0][0],dfy=B->bases[index-c0][1],dfz=B->bases[index-c0][2];
                    Coord dgx=-dfx,dgy=-dfy,dgz=-dfz;
                    for ( unsigned int i = 0; i < 8; i++ ) ref[i]=cubes[index-c0][i];
                    w[0]=gx*gy*gz;
                    w[1]=fx*gy*gz;
                    w[2]=fx*fy*gz;
                    w[3]=gx*fy*gz;
                    w[4]=gx*gy*fz;
                    w[5]=fx*gy*fz;
                    w[6]=fx*fy*fz;
                    w[7]=gx*fy*fz;

                    if(dw)
                    {
                        (*dw)[0]=dgy*gz*gx + dgz*gx*gy + dgx*gy*gz;
                        (*dw)[1]=dgy*gz*fx + dgz*fx*gy + dfx*gy*gz;
                        (*dw)[2]=dfy*gz*fx + dgz*fx*fy + dfx*fy*gz;
                        (*dw)[3]=dfy*gz*gx + dgz*gx*fy + dgx*fy*gz;
                        (*dw)[4]=dgy*fz*gx + dfz*gx*gy + dgx*gy*fz;
                        (*dw)[5]=dgy*fz*fx + dfz*fx*gy + dfx*gy*fz;
                        (*dw)[6]=dfy*fz*fx + dfz*fx*fy + dfx*fy*fz;
                        (*dw)[7]=dfy*fz*gx + dfz*gx*fy + dgx*fy*fz;
                    }
                    if(ddw)
                    {
                        (*ddw)[0]=B->covs(dgy,dgz)*gx + B->covs(dgz,dgx)*gy + B->covs(dgx,dgy)*gz;
                        (*ddw)[1]=B->covs(dgy,dgz)*fx + B->covs(dgz,dfx)*gy + B->covs(dfx,dgy)*gz;
                        (*ddw)[2]=B->covs(dfy,dgz)*fx + B->covs(dgz,dfx)*fy + B->covs(dfx,dfy)*gz;
                        (*ddw)[3]=B->covs(dfy,dgz)*gx + B->covs(dgz,dgx)*fy + B->covs(dgx,dfy)*gz;
                        (*ddw)[4]=B->covs(dgy,dfz)*gx + B->covs(dfz,dgx)*gy + B->covs(dgx,dgy)*fz;
                        (*ddw)[5]=B->covs(dgy,dfz)*fx + B->covs(dfz,dfx)*gy + B->covs(dfx,dgy)*fz;
                        (*ddw)[6]=B->covs(dfy,dfz)*fx + B->covs(dfz,dfx)*fy + B->covs(dfx,dfy)*fz;
                        (*ddw)[7]=B->covs(dfy,dfz)*gx + B->covs(dfz,dgx)*fy + B->covs(dgx,dfy)*fz;
                    }
                }

            }

            if ( index==-1 )
            {
                B->sout<<"point "<<childPosition<<" outside"<<B->sendl;
                ref.resize(0);  w.resize(0);
                if(dw) dw->resize(0);
                if(ddw) ddw->resize(0);
            }
        }
    };




    // 2D space
    template<class DUMMY>
    struct InternalShapeFunction<2,DUMMY>
    {
        static void getBasisFrom2DElements( Basis& b, const Coord& p0, const Coord& p1, const Coord& p2 )
        {
            b[0] = p1 - p0;
            b[1] = p2 - p0;
        }


        static double getDistanceTriangle( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( -(Real)0.01,v[0]+v[1]-(Real)1. ) );
        }
        static double getDistanceQuad( const Coord& v )
        {
            return std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-(Real)1.,v[0]-(Real)1. ), -(Real)0.01 ) );
        }


        static void init( BarycentricShapeFunction<ShapeFunctionTypes_>* B )
        {
            helper::ReadAccessor<Data<helper::vector<Coord> > > parent(B->f_position);
            if(!parent.size()) { B->serr<<"Parent nodes not found"<<B->sendl; return; }

            const Topo::SeqTriangles& triangles = B->parentTopology->getTriangles();
            const Topo::SeqQuads& quads = B->parentTopology->getQuads();
            const Topo::SeqEdges& edges = B->parentTopology->getEdges();

            if ( triangles.empty() && quads.empty() )
            {
                if ( edges.empty() ) return;

                //no 2D elements -> map on 1D elements
                B->f_nbRef.setValue(2);
                B->bases.resize ( edges.size() );
                for (unsigned int e=0; e<edges.size(); e++ )
                {
                    Coord V12 = ( parent[edges[e][1]]-parent[edges[e][0]] );
                    B->bases[e][0] = V12/V12.norm2();
                }
            }
            else
            {
                // map on 2D elements
                if(quads.size()) B->f_nbRef.setValue(4);
                else B->f_nbRef.setValue(3);
                B->bases.resize ( triangles.size() +quads.size() );
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    Basis m,mt;
                    getBasisFrom2DElements( m, parent[triangles[t][0]], parent[triangles[t][1]], parent[triangles[t][2]] );
                    mt.transpose ( m );
                    B->bases[t].invert ( mt );
                }
                int c0 = triangles.size();
                for ( unsigned int c = 0; c < quads.size(); c++ )
                {
                    Basis m,mt;
                    getBasisFrom2DElements( m, parent[quads[c][0]], parent[quads[c][1]], parent[quads[c][3]] );
                    mt.transpose ( m );
                    B->bases[c0+c].invert ( mt );
                }
            }
        }

        static void computeShapeFunction( const BarycentricShapeFunction<ShapeFunctionTypes_>* B, const Coord& childPosition, Cell &index, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)
        {
            // resize input
            unsigned int nbRef=B->f_nbRef.getValue();
            ref.resize(nbRef); ref.fill(0);
            w.resize(nbRef); w.fill(0);
            if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
            if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].clear(); }

            // get parent topology and nodes
            if(!B->parentTopology) return;

            helper::ReadAccessor<Data<helper::vector<Coord> > > parent(B->f_position);
            if(!parent.size()) return;

            const Topo::SeqTriangles& triangles = B->parentTopology->getTriangles();
            const Topo::SeqQuads& quads = B->parentTopology->getQuads();
            const Topo::SeqEdges& edges = B->parentTopology->getEdges();

            // compute barycentric weights by projection in cell basis
            index = -1;
            double distance = -B->f_tolerance.getValue();
            Coord coefs;

            if ( triangles.empty() && quads.empty() )
            {
                if ( edges.empty() ) return;
                //no 2D elements -> map on 1D elements
                for ( unsigned int i = 0; i < edges.size(); i++ )
                    if(cell==-1 || cell==(int)i)
                    {
                        Coord v = B->bases[i] * ( childPosition - parent[edges[i][0]] );
                        double d = std::max ( -v[0], v[0]-(Real)1. );
                        if ( d<=distance ) { coefs = v; distance = d; index = i; }
                    }
                if ( index!=-1 ) // addPointInLine
                {
                    ref[0]=edges[index][0]; ref[1]=edges[index][1];
                    // weights
                    w[0]=(Real)1.-coefs[0]; w[1]=coefs[0];
                    if(dw) {  (*dw)[0]=-B->bases[index][0]; (*dw)[1]=B->bases[index][0]; }
                }
            }
            else
            {
                // map on 2D elements
                for ( unsigned int i = 0; i < triangles.size(); i++ )
                    if(cell==-1 || cell==(int)i)
                    {
                        Coord v = B->bases[i] * ( childPosition - parent[triangles[i][0]] );
                        double d = getDistanceTriangle( v );
                        if ( d<=distance ) { coefs = v; distance = d; index = i; }
                    }
                int c0 = triangles.size();
                for ( unsigned int i = 0; i < quads.size(); i++ )
                    if(cell==-1 || cell==(int)(i+c0))
                    {
                        Coord v = B->bases[c0+i] * ( childPosition - parent[quads[i][0]] );
                        double d = getDistanceQuad( v );
                        if ( d<=distance ) { coefs = v; distance = d; index = c0+i; }
                    }
                if ( index!=-1 && index<c0 ) // addPointInTriangle
                {
                    ref[0]=triangles[index][0];                          ref[1]=triangles[index][1];  ref[2]=triangles[index][2];
                    // weights
                    w[0]=(Real)1.-coefs[0]-coefs[1];                     w[1]=coefs[0];               w[2]=coefs[1];
                    if(dw) {  (*dw)[0]=-B->bases[index][0]-B->bases[index][1]; (*dw)[1]=B->bases[index][0];    (*dw)[2]=B->bases[index][1]; }
                }
                else if ( index!=-1 ) // addPointInQuad
                {
                    Real fx=coefs[0],fy=coefs[1];
                    Real gx=(Real)1.-fx,gy=(Real)1.-fy;
                    Gradient dfx=B->bases[index-c0][0],dfy=B->bases[index-c0][1];
                    Gradient dgx=-dfx,dgy=-dfy;
                    for ( unsigned int i = 0; i < 4; i++ ) ref[i]=quads[index-c0][i];
                    // weights
                    w[0]=gx*gy; w[1]=fx*gy; w[2]=fx*fy; w[3]=gx*fy;
                    if(dw)
                    {
                        (*dw)[0]=dgx*gy+dgy*gx;
                        (*dw)[1]=dfx*gy+dgy*fx;
                        (*dw)[2]=dfx*fy+dfy*fx;
                        (*dw)[3]=dgx*fy+dfy*gx;
                    }
                    if(ddw)
                    {
                        (*ddw)[0]=B->covs(dgx,dgy);
                        (*ddw)[1]=B->covs(dfx,dgy);
                        (*ddw)[2]=B->covs(dfx,dfy);
                        (*ddw)[3]=B->covs(dgx,dfy);
                    }
                }
            }

            if ( index==-1 )
            {
                B->sout<<"point "<<childPosition<<" outside"<<B->sendl;
                ref.resize(0);  w.resize(0);
                if(dw) dw->resize(0);
                if(ddw) ddw->resize(0);
            }
        }


    };


public:
    // computes v1.v2^T + v2.v1^T
    inline Hessian covs(const Gradient& v1, const Gradient& v2) const
    {
        Hessian res;
        for ( int i = 0; i < Hessian::nbLines; ++i)
            for ( int j = i; j < Hessian::nbCols; ++j)
                res(i,j) = res(j,i) = v1[i] * v2[j] + v2[i] * v1[j];
        return res;
    }




    /// adapted from BarycentricMapperMeshTopology::init()
    virtual void init()
    {
        Inherit::init();

        if( !parentTopology )
        {
            Topo* topo;
            this->getContext()->get(topo,core::objectmodel::BaseContext::SearchUp);
            if(!topo) { serr<<"MeshTopology not found"<<sendl; return; }
            parentTopology.set(topo);
        }

        InternalShapeFunction<spatial_dimensions>::init( this );
    }

protected:
    BarycentricShapeFunction()
        : Inherit()
        , parentTopology(BaseLink::InitLink< BarycentricShapeFunction<ShapeFunctionTypes_> >(this, "parentTopology", ""))
        , f_tolerance(initData(&f_tolerance,(Real)-1.0,"tolerance","minimum weight (allows for mapping outside elements)"))
        , cellIndex(-1)
    {
    }

    virtual ~BarycentricShapeFunction()
    {

    }






};







}
}
}


#endif
