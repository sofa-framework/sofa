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

#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F331(type)  DefGradientTypes<3,3,0,type>
#define F332(type)  DefGradientTypes<3,3,1,type>
#define E221(type)  StrainTypes<2,2,0,type>
#define E331(type)  StrainTypes<3,3,0,type>
#define E332(type)  StrainTypes<3,3,1,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////
template<typename Real>
static Eigen::Matrix<Real,6,9,Eigen::RowMajor> assembleJ(const  Mat<3,3,Real>& f) // 3D
{
    static const unsigned int spatial_dimensions = 3;
    static const unsigned int material_dimensions = 3;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
    JBlock J=JBlock::Zero();
    for( unsigned int k=0; k<spatial_dimensions; k++ )
        for(unsigned int j=0; j<material_dimensions; j++)
            J(j,j+material_dimensions*k)=f[k][j];
    for( unsigned int k=0; k<spatial_dimensions; k++ )
    {
        J(3,material_dimensions*k+1)=J(5,material_dimensions*k+2)=f[k][0];
        J(3,material_dimensions*k)=J(4,material_dimensions*k+2)=f[k][1];
        J(5,material_dimensions*k)=J(4,material_dimensions*k+1)=f[k][2];
    }
    return J;
}

template<typename Real>
static Eigen::Matrix<Real,3,4,Eigen::RowMajor> assembleJ(const  Mat<2,2,Real>& f) // 2D
{
    static const unsigned int spatial_dimensions = 2;
    static const unsigned int material_dimensions = 2;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
    JBlock J=JBlock::Zero();
    for( unsigned int k=0; k<spatial_dimensions; k++ )
        for(unsigned int j=0; j<material_dimensions; j++)
            J(j,j+material_dimensions*k)=f[k][j];
    for( unsigned int k=0; k<spatial_dimensions; k++ )
    {
        J(material_dimensions,material_dimensions*k+1)=f[k][0];
        J(material_dimensions,material_dimensions*k)=f[k][1];
    }
    return J;
}

template<typename Real>
void  computeQR( Mat<3,3,Real> &r, const  Mat<3,3,Real>& f)
{
    Vec<3,Real> edgex(f[0][0],f[1][0],f[2][0]);
    Vec<3,Real> edgey(f[0][1],f[1][1],f[2][1]);
    helper::getRotation(r,edgex,edgey);
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
    typedef typename Inherit::KBlock KBlock;
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
            helper::polarDecomposition(data.getF(), R, strainmat);
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
        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += R*VoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        eB.template block(0,spatial_dimensions,strain_size,frame_size) = assembleJ(R);
        return B;
    }


    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        KBlock K = KBlock();
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};



////////////////////////////////////////////////////////////////////////////////////
//////  F321 -> E221
////////////////////////////////////////////////////////////////////////////////////

//template<class InReal,class OutReal>
//class CorotationalStrainJacobianBlock< F321(InReal) , E221(OutReal) > :
//        public  BaseJacobianBlock< F321(InReal) , E221(OutReal) >
//{
//public:
//    typedef F321(InReal) In;
//    typedef E221(OutReal) Out;

//    typedef BaseJacobianBlock<In,Out> Inherit;
//    typedef typename Inherit::InCoord InCoord;
//    typedef typename Inherit::InDeriv InDeriv;
//    typedef typename Inherit::OutCoord OutCoord;
//    typedef typename Inherit::OutDeriv OutDeriv;
//    typedef typename Inherit::MatBlock MatBlock;
//    typedef typename Inherit::KBlock KBlock;
//    typedef typename Inherit::Real Real;

//    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
//    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
//    enum { material_dimensions = In::material_dimensions };
//    enum { spatial_dimensions = In::spatial_dimensions };
//    enum { strain_size = Out::strain_size };
//    enum { frame_size = spatial_dimensions*material_dimensions };

//    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

//    /**
//    Mapping:   \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
//    where:  R/D are the rotational/skew symmetric parts of F=RD
//    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
//    */

//    static const bool constantJ=false;

//    Affine R;   ///< =  store rotational part of deformation gradient to compute J
//    unsigned int decompositionMethod;

//    void addapply( OutCoord& result, const InCoord& data32 )
//    {

//        // compute a 2*2 matrix based on the 3*2 matrix, in the plane of the  3-vectors.
//        F221 data;
//        Vec<3,Real> e0, e1, e2;
//        for(unsigned i=0; i<3; i++){
//            e0[i]=data32[i];
//            e1[i]=data32[i+3];
//        }
//        Vec<2,Real> f0(e0.norm(),0);
//        e0.normalize();
//        e2=cross(e0,e1);
//        e2.normalize();
//        Vec<3,Real> ee1=cross(e2,e0);
//        Vec<2,Real> f1(e1*e0,e1*ee1);
//        for(unsigned i=0; i<2; i++){
//            data[i][0]=f0[i];
//            data[i][1]=f1[i];
//        }

//        StrainMat strainmat;
//        if(decompositionMethod==0)      // polar
//            helper::polarDecomposition(data.getF(), R, strainmat);
//        else if(decompositionMethod==1)   // large (by QR)
//        {
//            computeQR(R,data.getF());
//            StrainMat T=R.transposed()*data.getF();
//            strainmat=(T+T.transposed())*(Real)0.5;
//        }
//        else if(decompositionMethod==2)   // small
//        {
//            strainmat=(data.getF()+data.getF().transposed())*(Real)0.5;
//            R.fill(0); for(unsigned int j=0;j<material_dimensions;j++) R[j][j]=(Real)1.;
//        }
//        for(unsigned int j=0;j<material_dimensions;j++) strainmat[j][j]-=(Real)1.;
//        result.getStrain() += MatToVoigt( strainmat );
//    }

//    void addmult( OutDeriv& result,const InDeriv& data )
//    {
//        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
//    }

//    void addMultTranspose( InDeriv& result, const OutDeriv& data )
//    {
//        result.getF() += R*VoigtToMat( data.getStrain() );
//    }

//    MatBlock getJ()
//    {
//        MatBlock B = MatBlock();
//        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
//        EigenMap eB(&B[0][0]);
//        // order 0
//        eB.template block(0,spatial_dimensions,strain_size,frame_size) = assembleJ(R);
//        return B;
//    }


//    // requires derivative of R. Not Yet implemented..
//    KBlock getK(const OutDeriv& /*childForce*/)
//    {
//        KBlock K = KBlock();
//        return K;
//    }
//    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
//    {
//    }
//};



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
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:
        - \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
        - \f$ E_k = [R^T F_k + F_k^T R ]/2  \f$
    where:
        - R/D are the rotational/skew symmetric parts of F=RD
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [R^T dF + dF^T R ]/2 \f$
        - \f$  dE_k = [R^T dF_k + dF_k^T R ]/2 \f$
      */

    static const bool constantJ=false;

    Affine R;   ///< =  store rotational part of deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat;
        if(decompositionMethod==0)      // polar
            helper::polarDecomposition(data.getF(), R, strainmat);
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

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T=R.transposed()*data.getGradientF(k);
            result.getStrainGradient(k) += MatToVoigt( (T+T.transposed())*(Real)0.5 );
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getStrainGradient(k) += MatToVoigt( R.multTranspose( data.getGradientF(k) ) );
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += R*VoigtToMat( data.getStrain() );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getGradientF(k) += R*VoigtToMat( data.getStrainGradient(k) );
        }
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        typedef Eigen::Matrix<Real,strain_size,frame_size,Eigen::RowMajor> JBlock;
        JBlock J = assembleJ(R);
        eB.template block(0,spatial_dimensions,strain_size,frame_size) = J;
        // order 1
        unsigned int offsetE=strain_size;
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            eB.template block(offsetE,spatial_dimensions+(k+1)*frame_size,strain_size,frame_size) = J;
            offsetE+=strain_size;
        }
        return B;
    }

    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        KBlock K;
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};


} // namespace defaulttype
} // namespace sofa



#endif
