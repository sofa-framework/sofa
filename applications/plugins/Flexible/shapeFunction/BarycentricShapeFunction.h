/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_BarycentricShapeFunction_H
#define FLEXIBLE_BarycentricShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include <sofa/core/topology/BaseMeshTopology.h>

#include <algorithm>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace shapefunction
{

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
/**
Barycentric shape functions are the barycentric coordinates of points inside cells (can be edges, triangles, quads, tetrahedra, hexahedra)
  */

template <class ShapeFunctionTypes_>
class BarycentricShapeFunction : public BaseShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BarycentricShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(BaseShapeFunction, ShapeFunctionTypes_));
    typedef BaseShapeFunction<ShapeFunctionTypes_> Inherit;

    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VCoord VCoord;
    enum {material_dimensions=Inherit::material_dimensions};
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;
    typedef typename Inherit::VRef VRef;

    typedef sofa::core::topology::BaseMeshTopology Topo;
    Topo* parentTopology;

    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::Hessian Hessian;
    typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
    typedef typename Inherit::VMaterialToSpatial VMaterialToSpatial;
    enum {spatial_dimensions=Inherit::spatial_dimensions};
    typedef Mat<spatial_dimensions,spatial_dimensions,Real> BasesType;
    sofa::helper::vector<BasesType> bases;

    /** @name orientation data */
    //@{
    Data< VCoord > f_orientation; // = rest deformation gradient orientation in each cell (Euler angles)
    Data< bool > f_useLocalOrientation;
    //@}

    void computeShapeFunction(const Coord& childPosition, MaterialToSpatial& M, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const int cell=-1)
    {
        M=MaterialToSpatial();
        for ( unsigned int i = 0; i < material_dimensions; i++ ) M[i][i]=(Real)1.; //identity

        // resize input
        unsigned int nbRef=this->f_nbRef.getValue();
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
        if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].clear(); }

        // get parent topology and nodes
        if(!this->parentTopology) return;

        helper::ReadAccessor<Data<vector<Coord> > > parent(this->f_position);
        if(!parent.size()) return;

        const Topo::SeqTetrahedra& tetrahedra = this->parentTopology->getTetrahedra();
        const Topo::SeqHexahedra& cubes = this->parentTopology->getHexahedra();
        const Topo::SeqTriangles& triangles = this->parentTopology->getTriangles();
        const Topo::SeqQuads& quads = this->parentTopology->getQuads();
        const Topo::SeqEdges& edges = this->parentTopology->getEdges();

        // compute barycentric weights by projection in cell basis
        int index = -1;
        double distance = 1;
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
                    Coord v = bases[i] * ( childPosition - parent[edges[i][0]] );
                    double d = std::max ( -v[0], v[0]-(Real)1. );
                    if ( d<=distance ) { coefs = v; distance = d; index = i; }
                }
                if ( index!=-1 ) // addPointInLine
                {
                    ref[0]=edges[index][0]; ref[1]=edges[index][1];
                    // local frame
                    BasesType R=getLocalFrame1D(parent[ref[0]],parent[ref[1]]); // orientation = local element frame (no user input for 1D elements)
                    for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=0; j<material_dimensions; j++) M[i][j]=R[i][j];
                    // weights
                    w[0]=(Real)1.-coefs[0]; w[1]=coefs[0];
                    if(dw) {  (*dw)[0]=-bases[index][0]; (*dw)[1]=bases[index][0]; }
                }
            }
            else
            {
                // no 3D elements -> map on 2D elements
                for ( unsigned int i = 0; i < triangles.size(); i++ )
                    if(cell==-1 || cell==(int)i)
                {
                    Coord v = bases[i] * ( childPosition - parent[triangles[i][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-(Real)0.01,v[0]+v[1]-(Real)1. ) );
                    if ( d<=distance ) { coefs = v; distance = d; index = i; }
                }
                int c0 = triangles.size();
                for ( unsigned int i = 0; i < quads.size(); i++ )
                    if(cell==-1 || cell==(int)(i+c0))
                {
                    Coord v = bases[c0+i] * ( childPosition - parent[quads[i][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-(Real)1.,v[0]-(Real)1. ),std::max ( v[2]-(Real)0.01,-v[2]-(Real)0.01 ) ) );
                    if ( d<=distance ) { coefs = v; distance = d; index = c0+i; }
                }
                if ( index!=-1 && index<c0 ) // addPointInTriangle
                {
                    ref[0]=triangles[index][0];                          ref[1]=triangles[index][1];  ref[2]=triangles[index][2];
                    // compute local orientation given user input and local element frame
                    BasesType U=getUserOrientation(index),R=getLocalFrame2D(parent[ref[0]],parent[ref[1]],parent[ref[2]]),A;
                    if(this->f_useLocalOrientation.getValue()) A=this->reorientGlobalOrientation(R*U,R);
                    else A=this->reorientGlobalOrientation(U,R);
                    for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=0; j<material_dimensions; j++) M[i][j]=A[i][j];
                    // weights
                    w[0]=(Real)1.-coefs[0]-coefs[1];                     w[1]=coefs[0];               w[2]=coefs[1];
                    if(dw) {  (*dw)[0]=-bases[index][0]-bases[index][1]; (*dw)[1]=bases[index][0];    (*dw)[2]=bases[index][1]; }
                }
                else if ( index!=-1 ) // addPointInQuad
                {
                    Real fx=coefs[0],fy=coefs[1];
                    Real gx=(Real)1.-fx,gy=(Real)1.-fy;
                    Gradient dfx=bases[index-c0][0],dfy=bases[index-c0][1];
                    Gradient dgx=-dfx,dgy=-dfy;
                    for ( unsigned int i = 0; i < 4; i++ ) ref[i]=quads[index-c0][i];
                    // compute local orientation given user input and local element frame
                    BasesType U=getUserOrientation(index-c0),R=getLocalFrame2D(parent[ref[0]],parent[ref[1]],parent[ref[3]]),A;
                    if(this->f_useLocalOrientation.getValue()) A=this->reorientGlobalOrientation(R*U,R);
                    else A=this->reorientGlobalOrientation(U,R);
                    for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=0; j<material_dimensions; j++) M[i][j]=A[i][j];
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
                        (*ddw)[0]=covs(dgx,dgy);
                        (*ddw)[1]=covs(dfx,dgy);
                        (*ddw)[2]=covs(dfx,dfy);
                        (*ddw)[3]=covs(dgx,dfy);
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
                Coord v = bases[i] * ( childPosition - parent[tetrahedra[i][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-(Real)1. ) );
                if ( d<=distance ) { coefs = v; distance = d; index = i; }
            }
            int c0 = tetrahedra.size();
            for ( unsigned int i = 0; i < cubes.size(); i++ )
                if(cell==-1 || cell==(int)(i+c0))
            {
                //Coord v = bases[c0+i] * ( childPosition - parent[cubes[i][0]] );  // for cuboid hexahedra
                Coord v; Coord ph[8];  for ( unsigned int j = 0; j < 8; j++ ) ph[j]=parent[cubes[i][j]]; this->computeHexaTrilinearWeights(v,ph,childPosition,1E-10); // for arbitrary hexahedra
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-(Real)1. ),std::max ( v[1]-1,v[2]-(Real)1. ) ) );
                if ( d<=distance ) { coefs = v; distance = d; index = c0+i; }
            }
            if ( index!=-1 && index<c0 ) // addPointInTet
            {
                ref[0]=tetrahedra[index][0];                                         ref[1]=tetrahedra[index][1];  ref[2]=tetrahedra[index][2];    ref[3]=tetrahedra[index][3];
                // compute local orientation given user input and local element frame
                BasesType U=getUserOrientation(index),A;
                if(this->f_useLocalOrientation.getValue()) A=getLocalFrame2D(parent[ref[0]],parent[ref[1]],parent[ref[2]])*U;
                else A=U;
                for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=0; j<material_dimensions; j++) M[i][j]=A[i][j];
                // weights
                w[0]=(Real)1.-coefs[0]-coefs[1]-coefs[2];                            w[1]=coefs[0];                w[2]=coefs[1];                  w[3]=coefs[2];
                if(dw) {  (*dw)[0]=-bases[index][0]-bases[index][1]-bases[index][2]; (*dw)[1]=bases[index][0];     (*dw)[2]=bases[index][1];       (*dw)[3]=bases[index][2];  }
            }
            else if ( index!=-1 ) // addPointInHex
            {
                Real fx=coefs[0],fy=coefs[1],fz=coefs[2];
                Real gx=(Real)1.-fx,gy=(Real)1.-fy,gz=(Real)1.-fz;
                Gradient dfx=bases[index-c0][0],dfy=bases[index-c0][1],dfz=bases[index-c0][2];
                Gradient dgx=-dfx,dgy=-dfy,dgz=-dfz;
                for ( unsigned int i = 0; i < 8; i++ ) ref[i]=cubes[index-c0][i];
                // compute local orientation given user input and local element frame
                BasesType U=getUserOrientation(index),A;
                if(this->f_useLocalOrientation.getValue()) A=getLocalFrame2D(parent[ref[0]],parent[ref[1]],parent[ref[4]])*U;
                else A=U;
                for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=0; j<material_dimensions; j++) M[i][j]=A[i][j];
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
                    (*ddw)[0]=covs(dgy,dgz)*gx + covs(dgz,dgx)*gy + covs(dgx,dgy)*gz;
                    (*ddw)[1]=covs(dgy,dgz)*fx + covs(dgz,dfx)*gy + covs(dfx,dgy)*gz;
                    (*ddw)[2]=covs(dfy,dgz)*fx + covs(dgz,dfx)*fy + covs(dfx,dfy)*gz;
                    (*ddw)[3]=covs(dfy,dgz)*gx + covs(dgz,dgx)*fy + covs(dgx,dfy)*gz;
                    (*ddw)[4]=covs(dgy,dfz)*gx + covs(dfz,dgx)*gy + covs(dgx,dgy)*fz;
                    (*ddw)[5]=covs(dgy,dfz)*fx + covs(dfz,dfx)*gy + covs(dfx,dgy)*fz;
                    (*ddw)[6]=covs(dfy,dfz)*fx + covs(dfz,dfx)*fy + covs(dfx,dfy)*fz;
                    (*ddw)[7]=covs(dfy,dfz)*gx + covs(dfz,dgx)*fy + covs(dgx,dfy)*fz;
                }
            }

        }

        //   if ( distance>0 ) sout<<"point "<<childPosition<<" outside"<<sendl;
    }

    // computes v1.v2^T + v2.v1^T
    inline Hessian covs(const Gradient& v1, const Gradient& v2)
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
            this->getContext()->get(parentTopology,core::objectmodel::BaseContext::SearchUp);
            if(!this->parentTopology) { serr<<"MeshTopology not found"<<sendl; return; }
        }

        helper::ReadAccessor<Data<vector<Coord> > > parent(this->f_position);
        if(!parent.size()) { serr<<"Parent nodes not found"<<sendl; return; }

        const Topo::SeqTetrahedra& tetrahedra = this->parentTopology->getTetrahedra();
        const Topo::SeqHexahedra& cubes = this->parentTopology->getHexahedra();
        const Topo::SeqTriangles& triangles = this->parentTopology->getTriangles();
        const Topo::SeqQuads& quads = this->parentTopology->getQuads();
        const Topo::SeqEdges& edges = this->parentTopology->getEdges();

        if ( tetrahedra.empty() && cubes.empty() )
        {
            if ( triangles.empty() && quads.empty() )
            {
                if ( edges.empty() ) return;

                //no 3D elements, nor 2D elements -> map on 1D elements
                this->f_nbRef.setValue(2);
                bases.resize ( edges.size() );
                for (unsigned int e=0; e<edges.size(); e++ )
                {
                    Coord V12 = ( parent[edges[e][1]]-parent[edges[e][0]] );
                    bases[e][0] = V12/V12.norm2();
                }
            }
            else
            {
                // no 3D elements -> map on 2D elements
                if(quads.size()) this->f_nbRef.setValue(4);
                else this->f_nbRef.setValue(3);
                bases.resize ( triangles.size() +quads.size() );
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    BasesType m,mt;
                    m[0] = parent[triangles[t][1]]-parent[triangles[t][0]];
                    m[1] = parent[triangles[t][2]]-parent[triangles[t][0]];
                    m[2] = cross ( m[0],m[1] );
                    mt.transpose ( m );
                    bases[t].invert ( mt );
                }
                int c0 = triangles.size();
                for ( unsigned int c = 0; c < quads.size(); c++ )
                {
                    BasesType m,mt;
                    m[0] = parent[quads[c][1]]-parent[quads[c][0]];
                    m[1] = parent[quads[c][3]]-parent[quads[c][0]];
                    m[2] = cross ( m[0],m[1] );
                    mt.transpose ( m );
                    bases[c0+c].invert ( mt );
                }
            }
        }
        else
        {
            // map on 3D elements
            if(cubes.size()) this->f_nbRef.setValue(8);
            else this->f_nbRef.setValue(4);
            bases.resize ( tetrahedra.size() +cubes.size() );
            for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
            {
                BasesType m,mt;
                m[0] = parent[tetrahedra[t][1]]-parent[tetrahedra[t][0]];
                m[1] = parent[tetrahedra[t][2]]-parent[tetrahedra[t][0]];
                m[2] = parent[tetrahedra[t][3]]-parent[tetrahedra[t][0]];
                mt.transpose ( m );
                bases[t].invert ( mt );
            }
            int c0 = tetrahedra.size();
            for ( unsigned int c = 0; c < cubes.size(); c++ )
            {
                BasesType m,mt;
                m[0] = parent[cubes[c][1]]-parent[cubes[c][0]];
                m[1] = parent[cubes[c][3]]-parent[cubes[c][0]];
                m[2] = parent[cubes[c][4]]-parent[cubes[c][0]];
                mt.transpose ( m );
                bases[c0+c].invert ( mt );
            }
        }
    }

protected:
    BarycentricShapeFunction()
        : Inherit()
        , parentTopology( NULL )
        , f_orientation(initData(&f_orientation,"orientation","input orientation (Euler angles) inside each cell"))
        , f_useLocalOrientation(initData(&f_useLocalOrientation,false,"useLocalOrientation","tells if orientations are defined in the local basis on each cell"))
    {
    }

    virtual ~BarycentricShapeFunction()
    {

    }

    // local coordinate systems
    inline BasesType getLocalFrame1D(const Coord &p1,const Coord &p2) const
    {
        BasesType R;
        R(0) = p2-p1; R(0).normalize();
        R(1) = Coord(1,0,0); if(dot(R(0),R(1))==1.) R(1) = Coord(0,1,0); if(dot(R(0),R(1))==1.) R(1) = Coord(0,0,1);
        R(2) = cross(R(0),R(1)); R(2).normalize();
        R(1) = cross(R(2),R(0));
        R.transpose();
        return R;
    }
    inline BasesType getLocalFrame2D(const Coord &p1,const Coord &p2,const Coord &p3) const
    {
        BasesType R;
        R(0) = p2-p1; R(0).normalize();
        R(1) = p3-p1;
        R(2) = cross(R(0),R(1)); R(2).normalize();
        R(1)=cross(R(2),R(0));
        R.transpose();
        return R;
    }

    // user provided orientation
    inline BasesType getUserOrientation(const unsigned int index) const
    {
    Coord orient; if(this->f_orientation.getValue().size()) orient=(this->f_orientation.getValue().size()>index)?this->f_orientation.getValue()[index]:this->f_orientation.getValue()[0];
    helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(orient * (Real)M_PI / (Real)180.0);
    BasesType R; q.toMatrix(R);
    return R;
    }

    // align user provided global orientation to 2D manifold (to required for 1D and 3D cells)
    inline BasesType reorientGlobalOrientation(const BasesType& Global, const BasesType& Local) const
    {
        BasesType R;
        R(2)=Local.transposed()(2); // third axis = face normal
        R(0)=Global.transposed()(0) - R(2)*dot(R(2),Global.transposed()(0)); R(0).normalize();
        R(1)= cross(R(2),R(0));
        R.transpose();
        return R;
    }


    /// computes w such that x= p0*(1-wx)*(1-wz)*(1-wy) + p1*wx*(1-wz)*(1-wy) + p5*wx*wz*(1-wy) + p4*(1-wx)*wz*(1-wy) + p3*(1-wx)*(1-wz)*wy + p2*wx*(1-wz)*wy + p6*wx*wz*wy + p7*(1-wx)*wz*wy
    /// using Newton method
    void computeHexaTrilinearWeights(Coord &w, const Coord p[8], const Coord &x,const Real &tolerance)
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

            Mat<3,3,Real> df;
            df[0] = - p[0]*g[2]*g[1] + p[1]*g[2]*g[1] + p[5]*w[2]*g[1] - p[4]*w[2]*g[1] - p[3]*g[2]*w[1] + p[2]*g[2]*w[1] + p[6]*w[2]*w[1] - p[7]*w[2]*w[1];
            df[1] = - p[0]*g[0]*g[2] - p[1]*w[0]*g[2] - p[5]*w[0]*w[2] - p[4]*g[0]*w[2] + p[3]*g[0]*g[2] + p[2]*w[0]*g[2] + p[6]*w[0]*w[2] + p[7]*g[0]*w[2];
            df[2] = - p[0]*g[0]*g[1] - p[1]*w[0]*g[1] + p[5]*w[0]*g[1] + p[4]*g[0]*g[1] - p[3]*g[0]*w[1] - p[2]*w[0]*w[1] + p[6]*w[0]*w[1] + p[7]*g[0]*w[1];

            Real det=determinant(df);
            if ( -MIN_DETERMINANT<=det && det<=MIN_DETERMINANT) { return; }
            Mat<3,3,Real> dfinv;
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
};


}
}
}


#endif
