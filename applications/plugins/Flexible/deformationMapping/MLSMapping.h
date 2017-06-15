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
#ifndef SOFA_COMPONENT_MAPPING_MLSMAPPING_H
#define SOFA_COMPONENT_MAPPING_MLSMAPPING_H

#include <Flexible/config.h>
#include "BaseDeformationMapping.h"
#include "BaseDeformationImpl.inl"
#include "MLSJacobianBlock_point.inl"
#include "MLSJacobianBlock_affine.inl"
#include "MLSJacobianBlock_rigid.inl"
#include "MLSJacobianBlock_quadratic.inl"

#ifdef __APPLE__
// a strange behaviour of the mac's linker requires to compile a few stuffs again
#include "BaseDeformationMapping.inl"
#endif

namespace sofa
{
namespace component
{
namespace mapping
{

/** Generic moving least squares mapping, from a variety of input types to a variety of output types.
*/

template <class TIn, class TOut>
class MLSMapping : public BaseDeformationMappingT<defaulttype::MLSJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::MLSJacobianBlock<TIn,TOut> BlockType;
    typedef BaseDeformationMappingT<BlockType> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VecCoord VecCoord;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InVecCoord InVecCoord;
    typedef typename Inherit::OutVecCoord OutVecCoord;

    typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::Hessian Hessian;
    typedef typename Inherit::VHessian VHessian;

    typedef defaulttype::MLSJacobianBlock<TIn,defaulttype::Vec3Types> PointMapperType;
    typedef defaulttype::DefGradientTypes<Inherit::spatial_dimensions, Inherit::material_dimensions, 0, Real> FType;
    typedef defaulttype::MLSJacobianBlock<TIn,FType> DeformationGradientMapperType;

    SOFA_CLASS(SOFA_TEMPLATE2(MLSMapping,TIn,TOut), SOFA_TEMPLATE(BaseDeformationMappingT,BlockType ));

    enum { spatial_dimensions = Inherit::spatial_dimensions };

protected:
    MLSMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
    {
    }

    virtual ~MLSMapping()     { }

    typedef defaulttype::InInfo<TIn> ininfo;
    typedef defaulttype::MLSInfo< ininfo::dim, ininfo::order, typename ininfo::Real > mlsinfo;
    typedef typename mlsinfo::basis basis;
    typedef typename mlsinfo::moment moment;

    ///< Compute the moment matrix \f$ M = sum w_i.xi*.xi*^T \f$ and its spatial derivatives (xi is the initial spatial position of node i)
    void computeMLSMatrices(moment& M, defaulttype::Vec<spatial_dimensions, moment>& dM, defaulttype::Mat<spatial_dimensions,spatial_dimensions, moment>& ddM,
                                const VRef& index, const InVecCoord& in, const VReal& w, const VGradient& dw, const VHessian& ddw)
    {
        M=moment();
        for(unsigned int k=0; k<spatial_dimensions; k++ ) dM(k)=moment();
        for(unsigned int i=0; i<spatial_dimensions; i++ ) for(unsigned int k=0; k<spatial_dimensions; k++ ) ddM(i,k)=moment();

        for(unsigned int j=0; j<index.size(); j++ )
        {
            moment XXT=mlsinfo::getCov(ininfo::getCenter(in[index[j]]));
            M += XXT * w[j];
            if(j<dw.size()) for(unsigned int k=0; k<spatial_dimensions; k++ ) dM(k)+=XXT * dw[j][k];
            if(j<ddw.size()) for(unsigned int i=0; i<spatial_dimensions; i++ ) for(unsigned int k=0; k<spatial_dimensions; k++ ) ddM(i,k)+=XXT * ddw[j](i,k);
        }
    }

    void invertMomentMatrix(moment& Minv,const moment& M)
    {
        // Minv.invert(M);
        static const unsigned int bdim=mlsinfo::bdim;
        Eigen::Matrix<Real,bdim,bdim>  eM;
        for(unsigned int k=0; k<bdim; k++ ) for(unsigned int l=0; l<bdim; l++ ) eM(k,l)=M(k,l);
        Eigen::Matrix<Real,bdim,bdim>  eMinv = eM.inverse();
        for(unsigned int k=0; k<bdim; k++ ) for(unsigned int l=0; l<bdim; l++ ) Minv(k,l)=eMinv(k,l);
    }


    ///< Compute the mls coordinates \f$ C = w.x*.x*^T M^{-1} p* \f$ and its spatial derivatives (p is the initial spatial position of a material point and x the initial spatial position of a node)
    void computeMLSCoordinates(basis& P, defaulttype::Vec<spatial_dimensions, basis>& dP, defaulttype::Mat<spatial_dimensions,spatial_dimensions, basis>& ddP,
                               const moment& Minv, const defaulttype::Vec<spatial_dimensions, moment>& dM, const defaulttype::Mat<spatial_dimensions,spatial_dimensions, moment>& ddM,
                               const Coord& p0, const InCoord& in, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        moment XXT=mlsinfo::getCov(ininfo::getCenter(in));

        basis P0 = mlsinfo::getBasis(defaulttype::InInfo<Coord>::getCenter(p0));
        defaulttype::Vec<spatial_dimensions, basis> dP0; for(unsigned int k=0; k<spatial_dimensions; k++ ) dP0[k]=mlsinfo::getBasisGradient(defaulttype::InInfo<Coord>::getCenter(p0),k);
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, basis> ddP0; for(unsigned int j=0; j<spatial_dimensions; j++ ) for(unsigned int k=0; k<spatial_dimensions; k++ ) ddP0(j,k)=mlsinfo::getBasisHessian(defaulttype::InInfo<Coord>::getCenter(p0),j,k);

        // P = w.x*.x*^T M^{-1} p0*
        P = XXT*(Minv * P0) * w;

        // dP(i) = x*.x*^T M^{-1} [p0*.dw(i) + w.dp0*(i) - w.dM(i) M^{-1} p0* ]
        for(unsigned int i=0; i<spatial_dimensions; i++ )
        {
            dP[i]= XXT*(Minv * (P0*dw[i] +  (dP0[i] - dM[i]*(Minv*P0))*w ));
        }

        // ddP(i,j) = x*.x*^T M^{-1} [  p0*.ddw(i,j) + w.ddp0*(i,j) + w(j).dp0*(i) + dp0*(j).dw(i)
        //                              - w *dM(i) M^{-1} p0*(j) - w * dM(j) M^{-1} dp0*(i)
        //                              + w.( dM(j) M^{-1} dM(i) + dM(i) M^{-1} dM(j) ).M^{-1} p0*
        //                              - dM(j) M^{-1} p0*.dw(i)     - dM(i) M^{-1} p0*.dw(j)
        //                              - w *ddM(i,j) M^{-1} p0* ]
        for(unsigned int i=0; i<spatial_dimensions; i++ ) for(unsigned int j=i; j<spatial_dimensions; j++ )
        {
            ddP(i,j)= XXT*(Minv * ( P0*ddw(i,j) + ddP0(i,j)*w + dP0[i]*dw[j] + dP0[j]*dw[i]
                                    -  dM[i]*(Minv*dP0[j])*w -  dM[j]*(Minv*dP0[i])*w
                                    +  dM[j]*(Minv*(dM[i]*(Minv*P0)))*w +  dM[i]*(Minv*(dM[j]*(Minv*P0)))*w
                                    - dM[j]*(Minv*P0)*dw[i]   - dM[i]*(Minv*P0)*dw[j]
                                    - ddM(i,j)*(Minv*P0)*w));
            if(j!=i) ddP(j,i)=ddP(i,j);
        }
    }




    virtual void initJacobianBlocks()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

        unsigned int size=this->f_pos0.getValue().size();
        this->jacobian.resize(size);

        moment M,Minv;
        defaulttype::Vec<spatial_dimensions, moment > dM;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, moment> ddM;

        basis P;
        defaulttype::Vec<spatial_dimensions, basis> dP;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, basis> ddP;

        for(unsigned int i=0; i<size; i++ )
        {
            unsigned int nbref=this->f_index.getValue()[i].size();
            this->jacobian[i].resize(nbref);

            computeMLSMatrices(M,dM,ddM,this->f_index.getValue()[i],in.ref(),this->f_w.getValue()[i],this->f_dw.getValue()[i],this->f_ddw.getValue()[i]);
            invertMomentMatrix(Minv,M);

            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                computeMLSCoordinates(P,dP,ddP,Minv,dM,ddM,this->f_pos0.getValue()[i],in[index],this->f_w.getValue()[i][j],this->f_dw.getValue()[i][j],this->f_ddw.getValue()[i][j]);
                this->jacobian[i][j].init(in[index] ,out[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],P,dP,ddP);
            }
        }
    }



    virtual void mapPosition(Coord& p,const Coord &p0, const VRef& ref, const VReal& w)
    {
        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

        PointMapperType mapper;

        // empty variables (not used in init)
        typename PointMapperType::OutCoord o(defaulttype::NOINIT);
        typename PointMapperType::MaterialToSpatial MtoS0(defaulttype::NOINIT);

        VGradient dw(1);
        VHessian ddw(1);

        moment M,Minv;
        defaulttype::Vec<spatial_dimensions, moment > dM;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, moment> ddM;
        basis P;
        defaulttype::Vec<spatial_dimensions, basis> dP;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, basis> ddP;

        computeMLSMatrices(M,dM,ddM,ref,in0.ref(),w,dw,ddw);
        invertMomentMatrix(Minv,M);

        p=Coord();
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            computeMLSCoordinates(P,dP,ddP,Minv,dM,ddM,p0,in0[index],w[j],dw[0],ddw[0]);
            mapper.init( in0[index],o,p0,MtoS0,P,dP,ddP);
            mapper.addapply(p,in[index]);
        }
    }

    virtual void mapDeformationGradient(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& MtoS, const VRef& ref, const VReal& w, const VGradient& dw)
    {
        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

        DeformationGradientMapperType mapper;

        // empty variables (not used in init)
        typename DeformationGradientMapperType::OutCoord o;
        VHessian ddw(1);

        moment M,Minv;
        defaulttype::Vec<spatial_dimensions, moment > dM;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, moment> ddM;
        basis P;
        defaulttype::Vec<spatial_dimensions, basis> dP;
        defaulttype::Mat<spatial_dimensions,spatial_dimensions, basis> ddP;

        computeMLSMatrices(M,dM,ddM,ref,in0.ref(),w,dw,ddw);
        invertMomentMatrix(Minv,M);

        typename DeformationGradientMapperType::OutCoord Fc;
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            computeMLSCoordinates(P,dP,ddP,Minv,dM,ddM,p0,in0[index],w[j],dw[j],ddw[0]);
            mapper.init( in0[index],o,p0,MtoS,P,dP,ddP);
            mapper.addapply(Fc,in[index]);
        }
        F=Fc.getF();
    }

};




} // namespace mapping
} // namespace component
} // namespace sofa

#endif

