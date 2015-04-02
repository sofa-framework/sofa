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
#ifndef SOFA_COMPONENT_MAPPING_LINEARMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_LINEARMULTIMAPPING_H

#include "../initFlexible.h"
#include "../deformationMapping/BaseDeformationMultiMapping.inl"
#include "../deformationMapping/LinearJacobianBlock_point.inl"
#include "../deformationMapping/LinearJacobianBlock_rigid.inl"
#include "../deformationMapping/LinearJacobianBlock_affine.inl"
#include "../deformationMapping/LinearJacobianBlock_quadratic.inl"
//#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/core/State.inl>

namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;


/** Generic linear mapping, from a variety of input types to a variety of output types.
*/


template <class TIn1, class TIn2, class TOut>
class LinearMultiMapping : public BaseDeformationMultiMappingT<defaulttype::LinearJacobianBlock<TIn1,TOut>, defaulttype::LinearJacobianBlock<TIn2,TOut> >
{
public:
    typedef defaulttype::LinearJacobianBlock<TIn1,TOut> BlockType1;
    typedef defaulttype::LinearJacobianBlock<TIn2,TOut> BlockType2;
    typedef BaseDeformationMultiMappingT<BlockType1,BlockType2> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VecCoord VecCoord;
    typedef typename Inherit::InVecCoord1 InVecCoord1;
    typedef typename Inherit::InVecCoord2 InVecCoord2;
    typedef typename Inherit::OutVecCoord OutVecCoord;

    typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;

    typedef defaulttype::StdVectorTypes<defaulttype::Vec<Inherit::spatial_dimensions,Real>,defaulttype::Vec<Inherit::spatial_dimensions,Real>,Real> VecSpatialDimensionType;
    typedef defaulttype::LinearJacobianBlock<TIn1,VecSpatialDimensionType> PointMapperType1;
    typedef defaulttype::LinearJacobianBlock<TIn2,VecSpatialDimensionType> PointMapperType2;
    typedef defaulttype::DefGradientTypes<Inherit::spatial_dimensions, Inherit::material_dimensions, 0, Real> FType;
    typedef defaulttype::LinearJacobianBlock<TIn1,FType> DeformationGradientMapperType1;
    typedef defaulttype::LinearJacobianBlock<TIn2,FType> DeformationGradientMapperType2;

    SOFA_CLASS(SOFA_TEMPLATE3(LinearMultiMapping,TIn1,TIn2,TOut), SOFA_TEMPLATE2(BaseDeformationMultiMappingT,BlockType1,BlockType2 ));

protected:
    LinearMultiMapping ()
        : Inherit ()
    {
    }

    virtual ~LinearMultiMapping()     { }


    virtual void mapPosition(Coord& p,const Coord &p0, const VRef& ref, const VReal& w)
    {
        helper::ReadAccessor<Data<InVecCoord1> > in10 (*this->fromModel1->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord1> > in1 (*this->fromModel1->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<InVecCoord2> > in20 (*this->fromModel2->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord2> > in2 (*this->fromModel2->read(core::ConstVecCoordId::position()));

        PointMapperType1 mapper1;
        PointMapperType2 mapper2;

        // empty variables (not used in init)
        typename PointMapperType1::OutCoord o(defaulttype::NOINIT);
        typename PointMapperType2::MaterialToSpatial M0(defaulttype::NOINIT);
        VGradient dw(1);
        VHessian ddw(1);

        p=Coord();
        size_t size1=this->getFromSize1();
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            if(index<size1)
            {
                mapper1.init( in10[index],o,p0,M0,w[j],dw[0],ddw[0]);
                mapper1.addapply(p,in1[index]);
            }
            else
            {
                mapper2.init( in20[index-size1],o,p0,M0,w[j],dw[0],ddw[0]);
                mapper2.addapply(p,in2[index-size1]);
            }
        }
    }

    virtual void mapDeformationGradient(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& M, const VRef& ref, const VReal& w, const VGradient& dw)
    {
        helper::ReadAccessor<Data<InVecCoord1> > in10 (*this->fromModel1->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord1> > in1 (*this->fromModel1->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<InVecCoord2> > in20 (*this->fromModel2->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord2> > in2 (*this->fromModel2->read(core::ConstVecCoordId::position()));

        DeformationGradientMapperType1 mapper1;
        DeformationGradientMapperType2 mapper2;

        // empty variables (not used in init)
        typename DeformationGradientMapperType1::OutCoord o;
        VHessian ddw(1);

        typename DeformationGradientMapperType1::OutCoord Fc;
        size_t size1=this->getFromSize1();
        for(unsigned int j=0; j<ref.size(); j++ )
        {
            unsigned int index=ref[j];
            if(index<size1)
            {
                mapper1.init( in10[index],o,p0,M,w[j],dw[j],ddw[0]);
                mapper1.addapply(Fc,in1[index]);
            }
            else
            {
                mapper2.init( in20[index-size1],o,p0,M,w[j],dw[j],ddw[0]);
                mapper2.addapply(Fc,in2[index-size1]);
            }
        }
        F=Fc.getF();
    }

    virtual void initJacobianBlocks()
    {
        helper::ReadAccessor<Data<InVecCoord1> > in1 (*this->fromModel1->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<InVecCoord2> > in2 (*this->fromModel2->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

        size_t size=this->f_pos0.getValue().size();

        this->jacobian1.resize(size);
        this->jacobian2.resize(size);
        size_t size1=this->getFromSize1();
        for(size_t i=0; i<size; i++ )
        {
            this->jacobian1[i].resize(0);
            this->jacobian2[i].resize(0);

            size_t nbref = this->f_index.getValue()[i].size();
            for(size_t j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                if(index<size1)
                {
                    BlockType1 b;
                    b.init( in1[index],out[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],this->f_w.getValue()[i][j],this->f_dw.getValue()[i][j],this->f_ddw.getValue()[i][j]);
                    this->jacobian1[i].push_back(b);
                }
                else
                {
                    BlockType2 b;
                    b.init( in2[index-size1],out[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],this->f_w.getValue()[i][j],this->f_dw.getValue()[i][j],this->f_ddw.getValue()[i][j]);
                    this->jacobian2[i].push_back(b);
                }
            }
        }
    }

};




} // namespace mapping
} // namespace component
} // namespace sofa

#endif

