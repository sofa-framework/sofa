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
#ifndef SOFA_COMPONENT_MAPPING_MLSMAPPING_H
#define SOFA_COMPONENT_MAPPING_MLSMAPPING_H

#include "../initFlexible.h"
#include "../deformationMapping/BaseDeformationMapping.inl"
#include "../deformationMapping/MLSJacobianBlock_point.inl"
//#include "../deformationMapping/MLSJacobianBlock_affine.inl"
//#include "../deformationMapping/MLSJacobianBlock_quadratic.inl"
#include <sofa/component/container/MechanicalObject.inl>
#include <sofa/core/State.inl>

namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;
using defaulttype::Mat;
using defaulttype::MatSym;



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
    typedef typename Inherit::VGradient VGradient;
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


    virtual void initJacobianBlocks()
    {
        std::cout<<"MLS: initJacobianBlocks"<<std::endl;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

        unsigned int size=this->f_pos0.getValue().size();
        this->jacobian.resize(size);

        static const unsigned int bdim=defaulttype::MLSInfo<InCoord>::bdim; // size of polynomial basis

        for(unsigned int i=0; i<size; i++ )
        {
            unsigned int nbref=this->f_index.getValue()[i].size();
            this->jacobian[i].resize(nbref);

            // compute moment matrix
            MatSym<bdim,Real> M;
            Vec<spatial_dimensions, MatSym<bdim,Real> > dM;

            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                MatSym<bdim,Real> XXT=defaulttype::MLSInfo<InCoord>::getCov(in[index]);
                M += XXT * this->f_w.getValue()[i][j];
                for(unsigned int k=0; k<spatial_dimensions; k++ ) dM[k]+=XXT * this->f_dw.getValue()[i][j][k];
            }

            MatSym<bdim,Real> Minv;
            // Minv.invert(M);

            Eigen::Matrix<Real,bdim,bdim>  eM;
            for(unsigned int k=0; k<bdim; k++ ) for(unsigned int l=0; l<bdim; l++ ) eM(k,l)=M(k,l);
            Eigen::Matrix<Real,bdim,bdim>  eMinv = eM.inverse();
            for(unsigned int k=0; k<bdim; k++ ) for(unsigned int l=0; l<bdim; l++ ) Minv(k,l)=eMinv(k,l);

            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                this->jacobian[i][j].init( in[index],out[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],this->f_w.getValue()[i][j],this->f_dw.getValue()[i][j],this->f_ddw.getValue()[i][j],Minv,dM);
            }
        }
    }



    virtual void mapPosition(Coord& /*p*/,const Coord &/*p0*/, const VRef& /*ref*/, const VReal& /*w*/)
    {
//        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
//        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

//        PointMapperType mapper;

//        // empty variables (not used in init)
//        typename PointMapperType::OutCoord o(defaulttype::NOINIT);
//        typename PointMapperType::MaterialToSpatial M0(defaulttype::NOINIT);
//        VGradient dw(1);
//        VHessian ddw(1);

//        p=Coord();
//        for(unsigned int j=0; j<ref.size(); j++ )
//        {
//            unsigned int index=ref[j];
//            mapper.init( in0[index],o,p0,M0,w[j],dw[0],ddw[0]);
//            mapper.addapply(p,in[index]);
//        }
    }

    virtual void mapDeformationGradient(MaterialToSpatial& /*F*/, const Coord &/*p0*/, const MaterialToSpatial& /*M*/, const VRef& /*ref*/, const VReal& /*w*/, const VGradient& /*dw*/)
    {
//        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
//        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

//        DeformationGradientMapperType mapper;

//        // empty variables (not used in init)
//        typename DeformationGradientMapperType::OutCoord o;
//        VHessian ddw(1);

//        typename DeformationGradientMapperType::OutCoord Fc;
//        for(unsigned int j=0; j<ref.size(); j++ )
//        {
//            unsigned int index=ref[j];
//            mapper.init( in0[index],o,p0,M,w[j],dw[j],ddw[0]);
//            mapper.addapply(Fc,in[index]);
//        }
//        F=Fc.getF();
    }

};




} // namespace mapping
} // namespace component
} // namespace sofa

#endif

