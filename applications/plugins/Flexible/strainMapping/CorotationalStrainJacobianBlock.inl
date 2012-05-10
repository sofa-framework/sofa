/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_CorotationalStrainJacobianBlock_INL
#define FLEXIBLE_CorotationalStrainJacobianBlock_INL

#include "../strainMapping/CorotationalStrainJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <sofa/helper/PolarDecompose.h>

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F331(type)  DefGradientTypes<3,3,0,type>
#define F332(type)  DefGradientTypes<3,3,1,type>
#define E331(type)  StrainTypes<3,3,0,type>
#define E332(type)  StrainTypes<3,3,1,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////

template<typename Real>
static Mat<6,9,Real> assembleJ(const  Mat<3,3,Real>& f) // 3D
{
    static const unsigned int spatial_dimensions = 3;
    static const unsigned int material_dimensions = 3;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    Mat<strain_size,spatial_dimensions*material_dimensions,Real> J;
    for( unsigned int k=0; k<spatial_dimensions; k++ )
        for(unsigned int j=0; j<material_dimensions; j++)
            J[j][j+material_dimensions*k]=f[k][j];
    for( unsigned int k=0; k<spatial_dimensions; k++ )
    {
        J[3][material_dimensions*k+1]=J[5][material_dimensions*k+2]=f[k][0];
        J[3][material_dimensions*k]=J[4][material_dimensions*k+2]=f[k][1];
        J[5][material_dimensions*k]=J[4][material_dimensions*k+1]=f[k][2];
    }
    return J;
}

template<typename Real>
static Mat<3,4,Real> assembleJ(const  Mat<2,2,Real>& f) // 2D
{
    static const unsigned int spatial_dimensions = 2;
    static const unsigned int material_dimensions = 2;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    Mat<strain_size,spatial_dimensions*material_dimensions,Real> J;
    for( unsigned int k=0; k<spatial_dimensions; k++ )
        for(unsigned int j=0; j<material_dimensions; j++)
            J[j][j+material_dimensions*k]=f[k][j];
    for( unsigned int k=0; k<spatial_dimensions; k++ )
    {
        J[material_dimensions][material_dimensions*k+1]=f[k][0];
        J[material_dimensions][material_dimensions*k]=f[k][1];
    }
    return J;
}


template<typename Real>
void  computeQR( Mat<3,3,Real> &r, const  Mat<3,3,Real>& f)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    typedef Vec<3,Real> Coord;
    Coord edgex(f[0][0],f[1][0],f[2][0]);
    edgex.normalize();

    Coord edgey(f[0][1],f[1][1],f[2][1]);

    Coord edgez = cross( edgex, edgey );
    edgez.normalize();

    edgey = cross( edgez, edgex );
    edgey.normalize();

    r[0][0] = edgex[0]; r[0][1] = edgey[0]; r[0][2] = edgez[0];
    r[1][0] = edgex[1]; r[1][1] = edgey[1]; r[1][2] = edgez[1];
    r[2][0] = edgex[2]; r[2][1] = edgey[2]; r[2][2] = edgez[2];
}

//////////////////////////////////////////////////////////////////////////////////
////  F331 -> E331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F331(InReal) , E331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , E331(OutReal) >
{
public:
    typedef F331(InReal) In;
    typedef E331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:   \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
    where:  R/D are the rotational/skew symmetric parts of F=RD
    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
    */

    static const bool constantJ=false;

    Affine R;   ///< =  store rotational part of deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;
        if(decompositionMethod==0)      // polar
            helper::polar_decomp(data.getF(), R, strainmat);
        else if(decompositionMethod==1)   // large (by QR)
        {
            computeQR(R,data.getF());
            StrainMat T=R.transposed()*data.getF();
            strainmat=(T+T.transposed())*(Real)0.5;
        }
        else if(decompositionMethod==2)   // small
        {
            strainmat=(data.getF()+data.getF().transposed())*(Real)0.5;
            R.fill(0); for(unsigned int j=0; j<material_dimensions; j++) R[j][j]=(Real)1.;
        }
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        StrainMat strainmat=R.multTranspose( data.getF() );
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += R*VoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        MatBlock B;
        Mat<strain_size,frame_size,Real> J = assembleJ(R);
        for(unsigned int j=0; j<strain_size; j++) memcpy(&B[j][spatial_dimensions],&J[j][0],frame_size*sizeof(Real)); // offset to account for spatialCoord of F
        return B;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  F332 -> E332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F332(InReal) , E332(OutReal) > :
    public  BaseJacobianBlock< F332(InReal) , E332(OutReal) >
{
public:
    typedef F332(InReal) In;
    typedef E332(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    /**
    Mapping:
        - \f$ E = [F^T.F - I ]/2  \f$
        - \f$ E_k = [(F_k^T.F + F^T.F_k ]/2  \f$
    where:
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [ F^T.dF + dF^T.F ]/2 \f$
        - \f$  dE_k = [ F_k^T.dF + dF^T.F_k + dF_k^T.F + F^T.dF_k]/2 \f$
      */

    static const bool constantJ=false;

    InCoord F;   ///< =  store deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {
        F=data;
        // order 0
        StrainMat strainmat=F.getF().multTranspose( F.getF() );
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=1.;
        strainmat*=(Real)0.5;
        result.getStrain() += MatToVoigt( strainmat );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            strainmat = F.getF().multTranspose( F.getGradientF(k) );
            result.getStrainGradient(k) += MatToVoigt( strainmat );
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        StrainMat strainmat=F.getF().multTranspose( data.getF() );
        result.getStrain() += MatToVoigt( strainmat );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            strainmat = F.getF().multTranspose( data.getGradientF(k) ) + F.getGradientF(k).multTranspose( data.getF() );
            result.getStrainGradient(k) += MatToVoigt( strainmat );
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        StrainMat strainmat=VoigtToMat( data.getStrain() );
        result.getF() += F.getF()*VoigtToMat( data.getStrain() );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            strainmat=VoigtToMat( data.getStrainGradient(k) );
            result.getF() += F.getGradientF(k)*strainmat;
            result.getGradientF(k) += F.getF()*strainmat;
        }
    }

    MatBlock getJ()
    {
        MatBlock B;
        // order 0
        Mat<strain_size,frame_size,Real> J = assembleJ(F.getF());
        for(unsigned int i=0; i<strain_size; i++)  memcpy(&B[i][spatial_dimensions],&J[i][0],frame_size*sizeof(Real));
        // order 1
        Vec<spatial_dimensions, Mat<strain_size,frame_size,Real> > Jgrad;
        for(unsigned int k=0; k<spatial_dimensions; k++) Jgrad[k]= assembleJ(F.getGradientF(k));

        unsigned int offsetE=strain_size;
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            for(unsigned int i=0; i<strain_size; i++)  memcpy(&B[i+offsetE][spatial_dimensions],&Jgrad[k][i][0],frame_size*sizeof(Real));
            for(unsigned int i=0; i<strain_size; i++)  memcpy(&B[i+offsetE][spatial_dimensions+(k+1)*frame_size],&J[i][0],frame_size*sizeof(Real));
            offsetE+=strain_size;
        }
        return B;
    }
};


} // namespace defaulttype
} // namespace sofa



#endif
