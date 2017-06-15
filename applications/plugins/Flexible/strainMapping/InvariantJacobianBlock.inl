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
#ifndef FLEXIBLE_InvariantJacobianBlock_INL
#define FLEXIBLE_InvariantJacobianBlock_INL

#include "../strainMapping/InvariantJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"
#include "../types/PolynomialBasis.h"

namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////

/// return dest = det(from).from^-T and det(from)
template<class real>
inline real getDeterminantGradient(Mat<3,3,real>& dest, const Mat<3,3,real>& from)
{
    real det=determinant(from);

    dest(0,0)= (from(1,1)*from(2,2) - from(2,1)*from(1,2));
    dest(0,1)= (from(1,2)*from(2,0) - from(2,2)*from(1,0));
    dest(0,2)= (from(1,0)*from(2,1) - from(2,0)*from(1,1));
    dest(1,0)= (from(2,1)*from(0,2) - from(0,1)*from(2,2));
    dest(1,1)= (from(2,2)*from(0,0) - from(0,2)*from(2,0));
    dest(1,2)= (from(2,0)*from(0,1) - from(0,0)*from(2,1));
    dest(2,0)= (from(0,1)*from(1,2) - from(1,1)*from(0,2));
    dest(2,1)= (from(0,2)*from(1,0) - from(1,2)*from(0,0));
    dest(2,2)= (from(0,0)*from(1,1) - from(1,0)*from(0,1));

    return det;
}

/// returns  d^2 det(from)/d from^2 = d( det(from).from^-T )/d from
template<class real>
static Mat<9,9,real> getDeterminantHessian(const Mat<3,3,real>& from)
{
    Mat<9,9,real> ret;

    ret(0,4)=ret(4,0)=from(2,2);    ret(0,5)=ret(5,0)=-from(2,1);    ret(0,7)=ret(7,0)=-from(1,2);    ret(0,8)=ret(8,0)=from(1,1);
    ret(1,3)=ret(3,1)=-from(2,2);   ret(1,5)=ret(5,1)=from(2,0);    ret(1,6)=ret(6,1)=from(1,2);    ret(1,8)=ret(8,1)=-from(1,0);
    ret(2,3)=ret(3,2)=from(2,1);    ret(2,4)=ret(4,2)=-from(2,0);    ret(2,6)=ret(6,2)=-from(1,1);    ret(2,7)=ret(7,2)=from(1,0);
    ret(3,2)=ret(2,3)=from(2,1);    ret(3,7)=ret(7,3)=from(0,2);    ret(3,8)=ret(8,3)=-from(0,1);
    ret(4,6)=ret(6,4)=-from(0,2);   ret(4,8)=ret(8,4)=from(0,0);
    ret(5,6)=ret(6,5)=from(0,1);    ret(5,7)=ret(7,5)=-from(0,0);
    ret(5,8)=ret(5,6)=from(0,1);
    return ret;
}

//template<class real>
//static void addMultDeterminantHessian(Mat<3,3,real>& out, const Mat<3,3,real>& in, const Mat<3,3,real>& from)
//{
//    out(0,0)+=from(2,2)*in(1,1)-from(2,1)*in(1,2)-from(1,2)*in(2,1)+from(1,1)*in(2,2);  out(0,1)+=-from(2,2)*in(1,0)+from(2,0)*in(1,2)+from(1,2)*in(2,0)-from(1,0)*in(2,2); out(0,2)+=from(2,1)*in(1,0)-from(2,0)*in(1,1)-from(1,1)*in(2,0)+from(1,0)*in(2,1);
//    out(1,0)+=-from(2,2)*in(0,1)+from(2,1)*in(0,2)+from(0,2)*in(2,1)-from(0,1)*in(2,2); out(1,1)+=from(2,2)*in(0,0)-from(2,0)*in(0,2)-from(0,2)*in(2,0)+from(0,0)*in(2,2);  out(1,2)+=-from(2,1)*in(0,0)+from(2,0)*in(0,1)+from(0,1)*in(2,0)-from(0,0)*in(2,1);
//    out(2,0)+=from(1,2)*in(0,1)-from(1,1)*in(0,2)-from(0,2)*in(1,1)+from(0,1)*in(1,2);  out(2,1)+=-from(1,2)*in(0,0)+from(1,0)*in(0,2)+from(0,2)*in(1,0)-from(0,0)*in(1,2); out(2,2)+=from(1,1)*in(0,0)-from(1,0)*in(0,1)-from(0,1)*in(1,0)+from(0,0)*in(1,1);
//}

/// returns  d^2 I1 /d from^2 = d( 2.*from )/d from
template<class real>
static Mat<9,9,real> getI1Hessian()
{
    Mat<9,9,real> ret;
    for ( unsigned int i = 0; i < 9; ++i)        ret(i,i)=(real)2.;
    return ret;
}

/// returns  d^2 I2 /d from^2 = d( from.(I1*Id - from^T.from) )/d from
template<class real>
static Mat<9,9,real> getI2Hessian(const Mat<3,3,real>& from)
{
    Mat<9,9,real> ret;
    ret(0,0) = from(1,1)*from(1,1)+from(2,1)*from(2,1)+from(1,2)*from(1,2)+from(2,2)*from(2,2); ret(0,1) = ret(1,0) = -from(1,0)*from(1,1)-from(2,0)*from(2,1);         ret(0,2) = ret(2,0) = -from(1,0)*from(1,2)-from(2,0)*from(2,2);          ret(0,3) = ret(3,0) = -from(1,1)*from(0,1)-from(1,2)*from(0,2); ret(0,4) = ret(4,0) = (real)2.*from(0,0)*from(1,1)-from(1,0)*from(0,1); ret(0,5) = ret(5,0) = (real)2.*from(0,0)*from(1,2)-from(1,0)*from(0,2); ret(0,6) = ret(6,0) = -from(2,1)*from(0,1)-from(2,2)*from(0,2); ret(0,7) = ret(7,0) = (real)2.*from(0,0)*from(2,1)-from(2,0)*from(0,1); ret(0,8) = ret(8,0) = (real)2.*from(0,0)*from(2,2)-from(2,0)*from(0,2);
    ret(1,1) = from(1,0)*from(1,0)+from(2,0)*from(2,0)+from(1,2)*from(1,2)+from(2,2)*from(2,2); ret(1,2) = ret(2,1) = -from(1,1)*from(1,2)-from(2,1)*from(2,2);         ret(1,3) = ret(3,1) = (real)2.*from(1,0)*from(0,1)-from(0,0)*from(1,1);  ret(1,4) = ret(4,1) = -from(1,0)*from(0,0)-from(1,2)*from(0,2); ret(1,5) = ret(5,1) = (real)2.*from(0,1)*from(1,2)-from(1,1)*from(0,2); ret(1,6) = ret(6,1) = (real)2.*from(2,0)*from(0,1)-from(0,0)*from(2,1); ret(1,7) = ret(7,1) = -from(2,0)*from(0,0)-from(2,2)*from(0,2); ret(1,8) = ret(8,1) = (real)2.*from(0,1)*from(2,2)-from(2,1)*from(0,2);
    ret(2,2) = from(1,0)*from(1,0)+from(2,0)*from(2,0)+from(1,1)*from(1,1)+from(2,1)*from(2,1); ret(2,3) = ret(3,2) = (real)2.*from(1,0)*from(0,2)-from(0,0)*from(1,2); ret(2,4) = ret(4,2) = (real)2.*from(1,1)*from(0,2)-from(0,1)*from(1,2);  ret(2,5) = ret(5,2) = -from(1,0)*from(0,0)-from(1,1)*from(0,1); ret(2,6) = ret(6,2) = (real)2.*from(2,0)*from(0,2)-from(0,0)*from(2,2); ret(2,7) = ret(7,2) = (real)2.*from(2,1)*from(0,2)-from(0,1)*from(2,2); ret(2,8) = ret(8,2) = -from(2,0)*from(0,0)-from(2,1)*from(0,1);
    ret(3,3) = from(0,1)*from(0,1)+from(2,1)*from(2,1)+from(0,2)*from(0,2)+from(2,2)*from(2,2); ret(3,4) = ret(4,3) = -from(0,0)*from(0,1)-from(2,0)*from(2,1);         ret(3,5) = ret(5,3) = -from(0,0)*from(0,2)-from(2,0)*from(2,2);          ret(3,6) = ret(6,3) = -from(2,1)*from(1,1)-from(2,2)*from(1,2); ret(3,7) = ret(7,3) = (real)2.*from(1,0)*from(2,1)-from(2,0)*from(1,1); ret(3,8) = ret(8,3) = (real)2.*from(1,0)*from(2,2)-from(2,0)*from(1,2);
    ret(4,4) = from(0,0)*from(0,0)+from(2,0)*from(2,0)+from(0,2)*from(0,2)+from(2,2)*from(2,2); ret(4,5) = ret(5,4) = -from(0,1)*from(0,2)-from(2,1)*from(2,2);         ret(4,6) = ret(6,4) = (real)2.*from(2,0)*from(1,1)-from(1,0)*from(2,1);  ret(4,7) = ret(7,4) = -from(2,0)*from(1,0)-from(2,2)*from(1,2); ret(4,8) = ret(8,4) = (real)2.*from(1,1)*from(2,2)-from(1,2)*from(2,1);
    ret(5,5) = from(0,0)*from(0,0)+from(2,0)*from(2,0)+from(0,1)*from(0,1)+from(2,1)*from(2,1); ret(5,6) = ret(6,5) = (real)2.*from(2,0)*from(1,2)-from(1,0)*from(2,2); ret(5,7) = ret(7,5) = (real)2.*from(1,2)*from(2,1)-from(1,1)*from(2,2);  ret(5,8) = ret(8,5) = -from(2,0)*from(1,0)-from(2,1)*from(1,1);
    ret(6,6) = from(0,1)*from(0,1)+from(1,1)*from(1,1)+from(0,2)*from(0,2)+from(1,2)*from(1,2); ret(6,7) = ret(7,6) = -from(0,0)*from(0,1)-from(1,0)*from(1,1);         ret(6,8) = ret(8,6) = -from(0,0)*from(0,2)-from(1,0)*from(1,2);
    ret(7,7) = from(0,0)*from(0,0)+from(1,0)*from(1,0)+from(0,2)*from(0,2)+from(1,2)*from(1,2); ret(7,8) = ret(8,7) = -from(0,1)*from(0,2)-from(1,1)*from(1,2);
    ret(8,8) = from(0,0)*from(0,0)+from(1,0)*from(1,0)+from(0,1)*from(0,1)+from(1,1)*from(1,1);
    ret*=(real)2.;
    return ret;
}


//////////////////////////////////////////////////////////////////////////////////
////  F331 -> I331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class InvariantJacobianBlock< F331(InReal) , I331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , I331(OutReal) >
{
public:
    typedef F331(InReal) In;
    typedef I331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient

    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;
    typedef Mat<frame_size,frame_size,Real> Hessian; ///< Matrix representing the Hessian of a scalar wrt. deformation gradient

    /**
    Mapping:
        - \f$ F -> [ I1 , I2, J ] \f$
    Jacobian:
        - \f$ J = [ dI1 , dI2 , dJ ] \f$
    Hessian:
        - \f$ H = [ ddI1 , ddI2 , ddJ ] \f$
    where:
        - \f$ I1 = trace(C) \f$ ,                           \f$ dI1 = trace(dF^T F + F^T dF ) = 2 sum F_i.dF_i \f$
        - \f$ I2 = [ ( trace(C)^2-trace(C^2) )/2 ]  \f$ ,   \f$ dI2 = 2 sum ( F(I1*Id - C) )_i dF_i \f$
        - \f$ J = det(F) \f$ ,                              \f$ dJ = J sum (F^-T)_i dF_i \f$
        - \f$ C=F^TF \f$ is the right Cauchy deformation tensor
    */

    static const bool constant=false;

    /// mapping parameters
    Frame dI1;
    Frame dI2;
    Frame dJ;

    Hessian dd1;
    Hessian dd2;
    Hessian ddJ;

    void addapply( OutCoord& result, const InCoord& data )
    {
        Frame F=data.getF();
        Real detF=getDeterminantGradient(dJ, F);

        StrainMat C=F.multTranspose( F );

        Real I1 = C[0][0] + C[1][1] + C[2][2];
        Real I2 = C[0][0]*(C[1][1] + C[2][2]) + C[1][1]*C[2][2] - C[0][1]*C[0][1] - C[0][2]*C[0][2] - C[1][2]*C[1][2];
        for(unsigned int j=0; j<material_dimensions; j++) C[j][j]-=I1;

        dI1=2.*F;
        dI2=-2.*F*C;

        // hessian
        ddJ = getDeterminantHessian(F);
        dd1 = getI1Hessian<Real>();
        dd2 = getI2Hessian(F);

        result.getStrain()[0]+= I1;
        result.getStrain()[1]+= I2;
        result.getStrain()[2]+= detF;

        //        std::cout<<"F="<<F<<std::endl;
        //        std::cout<<"dd1="<<dd1<<std::endl;
        //        std::cout<<"dd2="<<dd2<<std::endl;
        //        std::cout<<"ddJ="<<ddJ<<std::endl;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        Real di1 =  scalarProduct(dI1,data.getF());
        Real di2 =  scalarProduct(dI2,data.getF());
        Real dj =   scalarProduct(dJ,data.getF());

        result.getStrain()[0] +=  di1;
        result.getStrain()[1] +=  di2;
        result.getStrain()[2] +=  dj;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += dI1*data.getStrain()[0];
        result.getF() += dI2*data.getStrain()[1];
        result.getF() += dJ*data.getStrain()[2];
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        for(unsigned int j=0; j<frame_size; j++)      B(0,j) +=  *(&dI1[0][0]+j);
        for(unsigned int j=0; j<frame_size; j++)      B(1,j) +=  *(&dI2[0][0]+j);
        for(unsigned int j=0; j<frame_size; j++)      B(2,j) +=  *(&dJ[0][0]+j);
        return B;
    }

    KBlock getK(const OutDeriv& childForce, bool /*stabilization*/=false)
    {
        KBlock K = KBlock();
        K=dd1*childForce.getStrain()[0]+dd2*childForce.getStrain()[1]+ddJ*childForce.getStrain()[2];
        return K;
    }
    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
    {
        Hessian H=dd1*childForce.getStrain()[0]+dd2*childForce.getStrain()[1]+ddJ*childForce.getStrain()[2];
        const Vec<frame_size,Real>& vdx = *reinterpret_cast<const Vec<frame_size,Real>*>(&dx.getF()[0][0]);
        Vec<frame_size,Real>& vdf = *reinterpret_cast<Vec<frame_size,Real>*>(&df.getF()[0][0]);
        vdf += H*vdx*kfactor;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  U331 -> I331
//////////////////////////////////////////////////////////////////////////////////

//template<class InReal,class OutReal>
//class InvariantJacobianBlock< U331(InReal) , I331(OutReal) > :
//    public  BaseJacobianBlock< U331(InReal) , I331(OutReal) >
//{
//public:
//    typedef U331(InReal) In;
//    typedef I331(OutReal) Out;

//    typedef BaseJacobianBlock<In,Out> Inherit;
//    typedef typename Inherit::InCoord InCoord;
//    typedef typename Inherit::InDeriv InDeriv;
//    typedef typename Inherit::OutCoord OutCoord;
//    typedef typename Inherit::OutDeriv OutDeriv;
//    typedef typename Inherit::MatBlock MatBlock;
//    typedef typename Inherit::KBlock KBlock;
//    typedef typename Inherit::Real Real;

//    /**
//    Mapping:
//        - non-deviatoric: \f$ U -> [ sqrt(I1) , sqrt(I2), J ] \f$
//        - deviatoric \f$ U -> [ sqrt(I1/J^{2/3}) , sqrt(I2/J^{4/3}), J ] \f$
//    Jacobian:
//        - non-deviatoric: \f$ J = [ dI1/(2.sqrt(I1)) , dI2/(2.sqrt(I2)) , dJ ] = [ Ui/sqrt(I1) dUi , Ui.(Uj^2+Uk^2)/(2.sqrt(I2)) dUi, dJ ] \f$
//        - deviatoric \f$ J = [ Ui/(sqrt(I1/J^{2/3}).J^{2/3}) dUi , Ui.(Uj^2+Uk^2)/(sqrt(I2/J^{4/3}).J^{4/3}) dUi , dJ ] \f$
//    where:
//        - \f$ I1 = U1^2+U2^2+U3^2 \f$ ,                      \f$
//        - \f$ I2 = U1^2.U2^2 + U2^2.U3^2 + U3^2.U1^2  \f$    \f$
//        - \f$ J = U1*U2*U3                                   \f$ , \f$ dJ/dUi = Uj.Uk \f$
//    */

//    static const Real MIN_DETERMINANT() {return 0.001;} ///< J is clamped to avoid undefined deviatoric expressions

//    static const bool constant=false;
//    bool deviatoric;

//    /// mapping parameters
//    MatBlock _J;


//    void addapply( OutCoord& result, const InCoord& data )
//    {
//        Real squareU[3] = { data.getStrain()[0]*data.getStrain()[0], data.getStrain()[1]*data.getStrain()[1], data.getStrain()[2]*data.getStrain()[2] };

//        Real I1 = squareU[0] + squareU[1] + squareU[2];
//        Real I2 = squareU[0]*squareU[1] + squareU[1]*squareU[2] + squareU[2]*squareU[0];
//        Real J = data.getStrain()[0]*data.getStrain()[1]*data.getStrain()[2];
//        if ( J <= MIN_DETERMINANT() ) J = MIN_DETERMINANT();   // CLAMP J

//        Real sqrtI1, sqrtI2;
//        Real denI1, denI2;

//        if( deviatoric )
//        {
//            Real J23 = pow(J,2.0/3.0);
//            Real J43 = pow(J,4.0/3.0);

//            I1 /= J23;
//            I2 /= J43;
//            sqrtI1 = sqrt(I1);
//            sqrtI2 = sqrt(I2);

//            denI1 = sqrtI1 * J23;
//            denI2 = sqrtI2 * J43;
//        }
//        else
//        {
//            sqrtI1 = sqrt(I1);
//            sqrtI2 = sqrt(I2);

//            denI1 = sqrtI1;
//            denI2 = 2*sqrtI2;
//        }

//        _J[0][0] = data.getStrain()[0]/denI1;
//        _J[0][1] = data.getStrain()[1]/denI1;
//        _J[0][2] = data.getStrain()[2]/denI1;

//        _J[1][0] = data.getStrain()[0] * ( squareU[1] + squareU[2] ) / denI2;
//        _J[1][1] = data.getStrain()[1] * ( squareU[2] + squareU[0] ) / denI2;
//        _J[1][2] = data.getStrain()[2] * ( squareU[0] + squareU[1] ) / denI2;

//        _J[2][0] = data.getStrain()[1] * data.getStrain()[2];
//        _J[2][1] = data.getStrain()[2] * data.getStrain()[0];
//        _J[2][2] = data.getStrain()[0] * data.getStrain()[1];

//        result.getStrain()[0] += sqrtI1;
//        result.getStrain()[1] += sqrtI2;
//        result.getStrain()[2] += J;
//    }

//    void addmult( OutDeriv& result,const InDeriv& data )
//    {
//        result.getStrain() += _J * data.getStrain();
//    }

//    void addMultTranspose( InDeriv& result, const OutDeriv& data )
//    {
//        result.getStrain() += _J.multTranspose( data.getStrain() );
//    }

//    MatBlock getJ()
//    {
//        return _J;
//    }

//    KBlock getK(const OutDeriv&, bool=false)
//    {
//        KBlock K = KBlock();
//        return K;
//    }
//    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
//    {
//    }
//};

} // namespace defaulttype
} // namespace sofa



#endif
