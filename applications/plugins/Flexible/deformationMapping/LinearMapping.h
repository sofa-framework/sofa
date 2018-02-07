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
#ifndef SOFA_COMPONENT_MAPPING_LINEARMAPPING_H
#define SOFA_COMPONENT_MAPPING_LINEARMAPPING_H

#include <Flexible/config.h>
#include "BaseDeformationMapping.h"
#include "BaseDeformationImpl.inl"
#include "LinearJacobianBlock_point.inl"
#include "LinearJacobianBlock_rigid.inl"
#include "LinearJacobianBlock_affine.inl"
#include "LinearJacobianBlock_quadratic.inl"


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


///Generic linear mapping, from a variety of input types to a variety of output types.
template <class TIn, class TOut>
class LinearMapping : public BaseDeformationMappingT<defaulttype::LinearJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::LinearJacobianBlock<TIn,TOut> BlockType;
    typedef BaseDeformationMappingT<BlockType> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VecCoord VecCoord;
    typedef typename Inherit::InVecCoord InVecCoord;
    typedef typename Inherit::InVecDeriv InVecDeriv;
    typedef typename Inherit::OutVecCoord OutVecCoord;

    typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::Hessian Hessian;
    typedef typename Inherit::VHessian VHessian;

    typedef defaulttype::StdVectorTypes<defaulttype::Vec<Inherit::spatial_dimensions,Real>,defaulttype::Vec<Inherit::spatial_dimensions,Real>,Real> VecSpatialDimensionType;
    typedef defaulttype::LinearJacobianBlock<TIn,VecSpatialDimensionType> PointMapperType;
    typedef defaulttype::DefGradientTypes<Inherit::spatial_dimensions, Inherit::material_dimensions, 0, Real> FType;
    typedef defaulttype::LinearJacobianBlock<TIn,FType> DeformationGradientMapperType;

    SOFA_CLASS(SOFA_TEMPLATE2(LinearMapping,TIn,TOut), SOFA_TEMPLATE(BaseDeformationMappingT,BlockType ));

protected:
    LinearMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
    {
    }

    virtual ~LinearMapping()     { }


public :
    virtual void mapPosition(Coord& p,const Coord &p0, const VRef& ref, const VReal& w)
    {
        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

        PointMapperType mapper;

        // empty variables (not used in init)
        typename PointMapperType::OutCoord o(defaulttype::NOINIT);
        typename PointMapperType::MaterialToSpatial M0(defaulttype::NOINIT);
        VGradient dw(1);
        VHessian ddw(1);

        p=Coord();
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            mapper.init( in0[index],o,p0,M0,w[j],dw[0],ddw[0]);
            mapper.addapply(p,in[index]);
        }
    }

    virtual void mapDeformationGradient(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& M, const VRef& ref, const VReal& w, const VGradient& dw)
    {
        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

        DeformationGradientMapperType mapper;

        // empty variables (not used in init)
        typename DeformationGradientMapperType::OutCoord o;
        VHessian ddw(1);

        typename DeformationGradientMapperType::OutCoord Fc;
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            mapper.init( in0[index],o,p0,M,w[j],dw[j],ddw[0]);
            mapper.addapply(Fc,in[index]);
        }
        F=Fc.getF();
    }

    virtual void mapDeformationGradientRate(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& M, const VRef& ref, const VReal& w, const VGradient& dw)
    {
        helper::ReadAccessor<Data<InVecCoord> > in0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecDeriv> > in (*this->fromModel->read(core::ConstVecDerivId::velocity()));

        DeformationGradientMapperType mapper;

        // empty variables (not used in init)
        typename DeformationGradientMapperType::OutCoord o;
        VHessian ddw(1);

        typename DeformationGradientMapperType::OutCoord Fc;
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            mapper.init( in0[index],o,p0,M,w[j],dw[j],ddw[0]);
            mapper.addmult(Fc,in[index]);
        }
        F=Fc.getF();
    }

    virtual void initJacobianBlocks()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

        unsigned int size=this->f_pos0.getValue().size();

        bool dw  = !this->f_dw.getValue().empty();
        bool ddw = !this->f_ddw.getValue().empty();
        bool F0  = !this->f_F0.getValue().empty();

        static const MaterialToSpatial FI = identity<MaterialToSpatial>();
        this->jacobian.resize(size);
        for(unsigned int i=0; i<size; i++ )
        {
            unsigned int nbref=this->f_index.getValue()[i].size();
            this->jacobian[i].resize(nbref);
            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                this->jacobian[i][j].init( in[index],out[i],this->f_pos0.getValue()[i],
                                           F0 ? this->f_F0.getValue()[i] : FI,
                                           this->f_w.getValue()[i][j],
                                           dw  ? this->f_dw.getValue()[i][j]  : Gradient(),
                                           ddw ? this->f_ddw.getValue()[i][j] : Hessian()
                                         );
            }
        }
    }

    virtual void initJacobianBlocks(const InVecCoord& inCoord, const OutVecCoord& outCoord)
    {
        if(this->f_printLog.getValue())
            std::cout<<this->getName()<< "::" << SOFA_CLASS_METHOD <<std::endl;

        unsigned int cSize = this->f_pos0.getValue().size();
        this->jacobian.resize(cSize);
        for(unsigned int i=0; i<cSize; i++ )
        {
            unsigned int nbref=this->f_index.getValue()[i].size();
            this->jacobian[i].resize(nbref);
            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                this->jacobian[i][j].init( inCoord[index],outCoord[i],this->f_pos0.getValue()[i], this->f_F0.getValue()[i],
                                           this->f_w.getValue()[i][j], this->f_dw.getValue()[i][j], this->f_ddw.getValue()[i][j] );
            }
        }
    }

};




} // namespace mapping
} // namespace component
} // namespace sofa

#endif

