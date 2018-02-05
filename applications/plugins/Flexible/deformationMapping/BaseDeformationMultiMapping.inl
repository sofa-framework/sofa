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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMultiMapping_INL
#define SOFA_COMPONENT_MAPPING_BaseDeformationMultiMapping_INL

#include "BaseDeformationMultiMapping.h"
#include "BaseDeformationImpl.inl"
#include <SofaBaseVisual/VisualModelImpl.h>
#include "../quadrature/BaseGaussPointSampler.h"
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/IndexOpenMP.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace sofa
{
namespace component
{
namespace mapping
{

template <class JacobianBlockType1,class JacobianBlockType2>
BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::BaseDeformationMultiMappingT ()
    : Inherit ()
    , f_shapeFunction_name(initData(&f_shapeFunction_name,"shapeFunction","name of shape function (optional)"))
    , _shapeFunction(NULL)
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , f_index1 ( initData ( &f_index1,"indices1","parent1 indices for each child" ) )
    , f_index2 ( initData ( &f_index2,"indices2","parent2 indices for each child" ) )
    , f_w ( initData ( &f_w,"weights","influence weights of the Dofs" ) )
    , f_dw ( initData ( &f_dw,"weightGradients","weight gradients" ) )
    , f_ddw ( initData ( &f_ddw,"weightHessians","weight Hessians" ) )
    , f_F0 ( initData ( &f_F0,"M","Linear transformations from material to 3d space" ) )
    , f_cell ( initData ( &f_cell,"cell","indices required by shape function in case of overlapping elements" ) )
    , assemble ( initData ( &assemble,false, "assemble","Assemble the matrices (Jacobian/Geometric Stiffness) or use optimized Jacobian/vector multiplications" ) )
    , f_pos0 ( initData ( &f_pos0,"restPosition","initial spatial positions of children" ) )
    , missingInformationDirty(true)
    , KdTreeDirty(true)
    , fromModel1(NULL)
    , fromModel2(NULL)
    , toModel(NULL)
    , triangles(0)
    , extTriangles(0)
    , extvertPosIdx(0)
    , showDeformationGradientScale(initData(&showDeformationGradientScale, (float)0.0, "showDeformationGradientScale", "Scale for deformation gradient display"))
    , showDeformationGradientStyle ( initData ( &showDeformationGradientStyle,"showDeformationGradientStyle","Visualization style for deformation gradients" ) )
    , showColorOnTopology ( initData ( &showColorOnTopology,"showColorOnTopology","Color mapping method" ) )
    , showColorScale(initData(&showColorScale, (float)1.0, "showColorScale", "Color mapping scale"))
    , d_geometricStiffness(initData(&d_geometricStiffness, 0u, "geometricStiffness", "0=no GS, 1=non symmetric, 2=symmetrized"))
    , d_parallel(initData(&d_parallel, false, "parallel", "use openmp parallelisation?"))
{
    helper::OptionsGroup methodOptions(3,"0 - None"
                                       ,"1 - trace(F^T.F)-3"
                                       ,"2 - sqrt(det(F^T.F))-1");
    methodOptions.setSelectedItem(0);
    showColorOnTopology.setValue(methodOptions);

    helper::OptionsGroup styleOptions(6,"0 - All axis"
                                      ,"1 - First axis"
                                      ,"2 - Second axis"
                                      ,"3 - Third axis"
                                      ,"4 - deformation"
                                      ,"5 - 1st piola stress" );
    styleOptions.setSelectedItem(0);
    showDeformationGradientStyle.setValue(styleOptions);
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::resizeOut()
{
    {
        // TODO this must be done before resizeOut() but is done again in Inherit::init();
        // also clean the numerous calls to apply
        this->maskFrom1.resize( this->fromModels1.size() );
        for( unsigned i=0 ; i<this->fromModels1.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateFrom = this->fromModels1[i]->toBaseMechanicalState()) this->maskFrom1[i] = &stateFrom->forceMask;
        this->maskFrom2.resize( this->fromModels2.size() );
        for( unsigned i=0 ; i<this->fromModels2.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateFrom = this->fromModels2[i]->toBaseMechanicalState()) this->maskFrom2[i] = &stateFrom->forceMask;
        this->maskTo.resize( this->toModels.size() );
        for( unsigned i=0 ; i<this->toModels.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateTo = this->toModels[i]->toBaseMechanicalState()) this->maskTo[i] = &stateTo->forceMask;
            else this->setNonMechanical();
    }

    helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

    helper::WriteOnlyAccessor<Data<VecCoord> > pos0 (this->f_pos0);
    helper::WriteOnlyAccessor<Data< VMaterialToSpatial > >  F0(this->f_F0);
    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update mapped spatial positions if needed for visualization

    size_t size;

    engine::BaseGaussPointSampler* sampler;
    this->getContext()->get(sampler,core::objectmodel::BaseContext::Local);
    bool restPositionSet=false;
    helper::ReadAccessor<Data< OutVecCoord > >  rest(*this->toModel->read(core::ConstVecCoordId::restPosition()));

    if(sampler) // retrieve initial positions from gauss point sampler (deformation gradient types)
    {
        size = sampler->getNbSamples();
        if(rest.size()==size && size!=1)  restPositionSet=true;

        this->toModel->resize(size);
        pos0.resize(size);  for(size_t i=0; i<size; i++) pos0[i]=sampler->getSamples()[i];
        F0.resize(size);
        if(restPositionSet)     // use custom rest positions defined in state (to set material directions or set residual deformations)
        {
            for(size_t i=0; i<rest.size(); ++i) F0[i]=OutDataTypesInfo<Out>::getF(rest[i]);
            sout<<rest.size()<<" rest positions imported "<<sendl;
        }
        else
        {
            for(size_t i=0; i<size; ++i) copy(F0[i],sampler->getTransforms()[i]);
        }
        sout<<size <<" gauss points imported"<<sendl;
    }
    else  // retrieve initial positions from children dofs (vec types)
    {
        size = out.size();
        pos0.resize(size);  for(size_t i=0; i<size; i++ )  Out::get(pos0[i][0],pos0[i][1],pos0[i][2],out[i]);
    }

    // init shape function
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    std::vector<BaseShapeFunction*> sf; context->get<BaseShapeFunction>(&sf,core::objectmodel::BaseContext::SearchUp);
    for(unsigned int i=0;i<sf.size();i++)
    {
        if(this->f_shapeFunction_name.isSet()) {if(this->f_shapeFunction_name.getValue().compare(sf[i]->getName()) == 0) _shapeFunction=sf[i];}
        else if(sf[i]->f_position.getValue().size() == this->fromModel1->getSize()+this->fromModel2->getSize()) _shapeFunction=sf[i];
    }

    if(0 != f_index.getValue().size() && pos0.size() == f_index.getValue().size() && f_w.getValue().size() == f_index.getValue().size()) // we already have the needed data, we directly use them
    {
        sout<<"using filled data"<<sendl;
    }
    else if (_shapeFunction) // if we do not have the needed data, and have a shape function, we use it to compute needed data (index, weights, etc.)
    {
        sout<<"found shape function "<<_shapeFunction->getName()<<sendl;
        helper::vector<mCoord> mpos0;
        mpos0.resize(pos0.size());
        for(size_t i=0; i<pos0.size(); ++i)  defaulttype::StdVectorTypes<mCoord,mCoord>::set( mpos0[i], pos0[i][0] , pos0[i][1] , pos0[i][2]);

        // interpolate weights at sample positions
        if(this->f_cell.getValue().size()==size) _shapeFunction->computeShapeFunction(mpos0,*this->f_index.beginWriteOnly(),*this->f_w.beginWriteOnly(),*this->f_dw.beginWriteOnly(),*this->f_ddw.beginWriteOnly(),this->f_cell.getValue());
        else _shapeFunction->computeShapeFunction(mpos0,*this->f_index.beginWriteOnly(),*this->f_w.beginWriteOnly(),*this->f_dw.beginWriteOnly(),*this->f_ddw.beginWriteOnly());
        this->f_index.endEdit();    this->f_w.endEdit();        this->f_dw.endEdit();        this->f_ddw.endEdit();
    }
    else // if the prerequisites are not fulfilled we print an error
    {
        serr << "ShapeFunction<"<<ShapeFunctionType::Name()<<"> component not found" << sendl;
    }

    // update indices for each parent type
    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index1 (this->f_index1);   wa_index1.resize(size);
    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index2 (this->f_index2);   wa_index2.resize(size);
    for(size_t i=0; i<size; i++ )
    {
        for(size_t j=0; j<this->f_index.getValue()[i].size(); j++ )
        {
            unsigned int index = this->f_index.getValue()[i][j];
            if(index<this->getFromSize1()) wa_index1[i].push_back(index);
            else wa_index2[i].push_back(index-this->getFromSize1());
        }
    }

    // init jacobians
    initJacobianBlocks();

    // clear forces
    if(this->toModel->write(core::VecDerivId::force())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  f(*this->toModel->write(core::VecDerivId::force())); for(size_t i=0;i<f.size();i++) f[i].clear(); }
    // clear velocities
    if(this->toModel->write(core::VecDerivId::velocity())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  vel(*this->toModel->write(core::VecDerivId::velocity())); for(size_t i=0;i<vel.size();i++) vel[i].clear(); }

    reinit();
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::resizeOut(const helper::vector<Coord>& position0, helper::vector<helper::vector<unsigned int> > index,helper::vector<helper::vector<Real> > w, helper::vector<helper::vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, helper::vector<helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0)
{
    {
        // TODO this must be done before resizeOut() but is done again in Inherit::init();
        // also clean the numerous calls to apply
        this->maskFrom1.resize( this->fromModels1.size() );
        for( unsigned i=0 ; i<this->fromModels1.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateFrom = this->fromModels1[i]->toBaseMechanicalState()) this->maskFrom1[i] = &stateFrom->forceMask;
        this->maskFrom2.resize( this->fromModels2.size() );
        for( unsigned i=0 ; i<this->fromModels2.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateFrom = this->fromModels2[i]->toBaseMechanicalState()) this->maskFrom2[i] = &stateFrom->forceMask;
        this->maskTo.resize( this->toModels.size() );
        for( unsigned i=0 ; i<this->toModels.size() ; ++i )
            if (core::behavior::BaseMechanicalState* stateTo = this->toModels[i]->toBaseMechanicalState()) this->maskTo[i] = &stateTo->forceMask;
            else this->setNonMechanical();
    }

    helper::WriteOnlyAccessor<Data<VecCoord> > pos0 (this->f_pos0);
    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update mapped spatial positions if needed for visualization

    size_t size = position0.size();

    // paste input values
    this->toModel->resize(size);
    pos0.resize(size);  for(size_t i=0; i<size; i++ )        pos0[i]=position0[i];

    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index (this->f_index);   wa_index.resize(size);  for(size_t i=0; i<size; i++ )    wa_index[i].assign(index[i].begin(), index[i].end());
    helper::WriteOnlyAccessor<Data<VecVReal > > wa_w (this->f_w);          wa_w.resize(size);  for(size_t i=0; i<size; i++ )    wa_w[i].assign(w[i].begin(), w[i].end());
    helper::WriteOnlyAccessor<Data<VecVGradient > > wa_dw (this->f_dw);    wa_dw.resize(size);  for(size_t i=0; i<size; i++ )    wa_dw[i].assign(dw[i].begin(), dw[i].end());
    helper::WriteOnlyAccessor<Data<VecVHessian > > wa_ddw (this->f_ddw);   wa_ddw.resize(size);  for(size_t i=0; i<size; i++ )    wa_ddw[i].assign(ddw[i].begin(), ddw[i].end());
    helper::WriteOnlyAccessor<Data<VMaterialToSpatial> > wa_F0 (this->f_F0);    wa_F0.resize(size);  for(size_t i=0; i<size; i++ )    for(size_t j=0; j<spatial_dimensions; j++ ) for(size_t k=0; k<material_dimensions; k++ )   wa_F0[i][j][k]=F0[i][j][k];

    sout<<size <<" custom gauss points imported"<<sendl;

    // init jacobians
    initJacobianBlocks();

    // update indices for each parent type
    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index1 (this->f_index1);   wa_index1.resize(size);
    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index2 (this->f_index2);   wa_index2.resize(size);
    for(size_t i=0; i<size; i++ )
    {
        for(size_t j=0; j<this->f_index.getValue()[i].size(); j++ )
        {
            unsigned int index = this->f_index.getValue()[i][j];
            if(index<this->getFromSize1()) wa_index1[i].push_back(index);
            else wa_index2[i].push_back(index-this->getFromSize1());
        }
    }

    // clear forces
    if(this->toModel->write(core::VecDerivId::force())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  f(*this->toModel->write(core::VecDerivId::force())); for(size_t i=0;i<f.size();i++) f[i].clear(); }
    // clear velocities
    if(this->toModel->write(core::VecDerivId::velocity())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  vel(*this->toModel->write(core::VecDerivId::velocity())); for(size_t i=0;i<vel.size();i++) vel[i].clear(); }

    reinit();
}


template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::init()
{
    if(!this->fromModel1 && this->getFromModels1().size())     this->fromModel1 = this->getFromModels1()[0];
    if(!this->fromModel2 && this->getFromModels2().size())     this->fromModel2 = this->getFromModels2()[0];
    if(!this->toModel && this->getToModels().size())           this->toModel = this->getToModels()[0];

    if ( !this->toModel->toBaseMechanicalState())  this->setNonMechanical();

    component::visualmodel::VisualModelImpl *visual;
    this->getContext()->get( visual, core::objectmodel::BaseContext::Local);
    if(visual) {this->extTriangles = &visual->getTriangles(); this->extvertPosIdx = &visual->m_vertPosIdx.getValue(); this->triangles=0; }
    else
    {
        core::topology::BaseMeshTopology *topo;
        this->getContext()->get( topo, core::objectmodel::BaseContext::Local);
        if(topo) {this->triangles = &topo->getTriangles();  this->extTriangles=0; }
    }

    baseMatrices.resize( 2 ); // just a wrapping for getJs()
    baseMatrices[0] = &eigenJacobian1;
    baseMatrices[1] = &eigenJacobian2;

    resizeOut();

    Inherit::init();

    // check that all children particles got a parent
    const VecVRef& indices = this->f_index.getValue();
    for (std::size_t i=0; i < indices.size(); ++i)
        if (indices[i].empty())
            serr << "Particle " << i << " has no parent" << sendl;
}

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::reinit()
{
    if(this->assemble.getValue()) { updateJ1(); updateJ2(); }

    // force apply
    // bg: do we need this ?
    //    apply(NULL, *this->toModel->write(core::VecCoordId::position()), *this->fromModel1->read(core::ConstVecCoordId::position()), *this->fromModel2->read(core::ConstVecCoordId::position()) );
    //    if(this->toModel->write(core::VecDerivId::velocity())) applyJ(NULL, *this->toModel->write(core::VecDerivId::velocity()), *this->fromModel1->read(core::ConstVecDerivId::velocity()), *this->fromModel2->read(core::ConstVecDerivId::velocity()));

    Inherit::reinit();
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::updateJ1()
{
    eigenJacobian1.resizeBlocks(jacobian1.size(),getFromSize1());

    const VecVRef& index1 = this->f_index1.getValue();

    for( size_t i=0 ; i<getToSize() ; ++i)
    {
        eigenJacobian1.beginBlockRow(i);
        for(size_t j=0; j<jacobian1[i].size(); j++)
            eigenJacobian1.createBlock( index1[i][j], jacobian1[i][j].getJ());
        eigenJacobian1.endBlockRow();
    }

    eigenJacobian1.compress();

//    maskedEigenJacobian1.resize(0,0);
}

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::updateJ2()
{
    eigenJacobian2.resizeBlocks(jacobian2.size(),getFromSize2());

    const VecVRef& index2 = this->f_index2.getValue();

    for( size_t i=0 ; i<getToSize() ; ++i)
    {
        eigenJacobian2.beginBlockRow(i);
        for(size_t j=0; j<jacobian2[i].size(); j++)
            eigenJacobian2.createBlock( index2[i][j], jacobian2[i][j].getJ());
        eigenJacobian2.endBlockRow();
    }

    eigenJacobian2.compress();

//    maskedEigenJacobian2.resize(0,0);
}

//template <class JacobianBlockType1,class JacobianBlockType2>
//void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::updateMaskedJ()
//{
//    size_t currentHash = this->maskTo[0]->getHash();
//    if( previousMaskHash!=currentHash )
//    {
//        previousMaskHash = currentHash;
//        maskedEigenJacobian1.resize(0,0);
//        maskedEigenJacobian2.resize(0,0);
//    }
//    if( !maskedEigenJacobian1.rows() )
//        this->maskTo[0]->maskedMatrix( maskedEigenJacobian1.compressedMatrix, eigenJacobian1.compressedMatrix, Out::deriv_total_size );
//    if( !maskedEigenJacobian2.rows() )
//        this->maskTo[0]->maskedMatrix( maskedEigenJacobian2.compressedMatrix, eigenJacobian2.compressedMatrix, Out::deriv_total_size );
//}


template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForceId*/ )
{
    if( (BlockType1::constant && BlockType2::constant) || !d_geometricStiffness.getValue() ) return;

    // TODO IMPLEMENT THIS SUCH AS EVERY BLOCK ARE ADDED AT THE RIGHT PLACE IN THE GLOBAL MATRIX
    // ANYWAY I AM NOT SURE IT CAN BE IMPLEMENTED THIS WAY (WITH INDEPENDANT BLOCKS)
    // OTHERWISE CROSS-TERMS WILL BE LOST

    serr<<"The generic implementation of updateK has not be updated to the new BaseMapping::getK API\n";

//    size_t size1 = this->getFromSize1();
//    K1.resizeBlocks(size1,size1);
//    vector<KBlock1> diagonalBlocks1; diagonalBlocks1.resize(size1);

//    if( !this->maskTo || !this->maskTo->isInUse() )
//    {
//        for(size_t i=0; i<jacobian1.size(); i++)
//        {
//            for(size_t j=0; j<jacobian1[i].size(); j++)
//            {
//                unsigned int index = this->f_index1.getValue()[i][j];
//                diagonalBlocks1[index] += jacobian1[i][j].getK(childForce[i]);
//            }
//        }
//    }
//    else
//    {
//        const helper::StateMask::InternalStorage &indices=this->maskTo->getEntries();
//        for (helper::StateMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
//        {
//            size_t i= ( size_t ) ( *it );
//            for(size_t j=0; j<jacobian1[i].size(); j++)
//            {
//                unsigned int index = this->f_index1.getValue()[i][j];
//                diagonalBlocks1[index] += jacobian1[i][j].getK(childForce[i]);
//            }
//        }
//    }

//    for(size_t i=0; i<size1; i++)
//    {
//        K1.beginBlockRow(i);
//        K1.createBlock(i,diagonalBlocks1[i]);
//        K1.endBlockRow();
//    }
//    K1.compress();
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord1>& dIn1, const Data<InVecCoord2>& dIn2)
{
    helper::ReadAccessor<Data<OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
    //    if(_sampler) if(_sampler->getNbSamples()!=outpos.size()) resizeOut();

    OutVecCoord& out = *dOut.beginWriteOnly();
    const InVecCoord1& in1 = dIn1.getValue();
    const InVecCoord2& in2 = dIn2.getValue();

    const VecVRef& index1 = this->f_index1.getValue();
    const VecVRef& index2 = this->f_index2.getValue();

#ifdef _OPENMP
#pragma omp parallel for if (this->d_parallel.getValue())
#endif
	for(sofa::helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian1.size(); i++)
    {
        out[i]=OutCoord();
        for(size_t j=0; j<jacobian1[i].size(); j++)
        {
            size_t index=index1[i][j];
            jacobian1[i][j].addapply(out[i],in1[index]);
        }
        for(size_t j=0; j<jacobian2[i].size(); j++)
        {
            size_t index=index2[i][j];
            jacobian2[i][j].addapply(out[i],in2[index]);
        }
    }

    dOut.endEdit();

    if(this->assemble.getValue())
    {
        if(!BlockType1::constant) eigenJacobian1.resize(0,0);
        if(!BlockType2::constant) eigenJacobian2.resize(0,0);
    };

    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update spatial positions of defo grads if needed for visualization
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv1>& dIn1, const Data<InVecDeriv2>& dIn2)
{
    if(this->assemble.getValue())
    {
        if( !eigenJacobian1.rows() ) updateJ1();
        if( !eigenJacobian2.rows() ) updateJ2();

//        if( this->maskTo[0]->isActivated() )
//        {
//            updateMaskedJ();
//            maskedEigenJacobian1.mult(dOut,dIn1);
//            maskedEigenJacobian2.mult(dOut,dIn2);
//        }
//        else
        {
            eigenJacobian1.mult(dOut,dIn1);
            eigenJacobian2.addMult(dOut,dIn2);
        }
    }
    else
    {
        OutVecDeriv& out = *dOut.beginWriteOnly();
        const InVecDeriv1& in1 = dIn1.getValue();
        const InVecDeriv2& in2 = dIn2.getValue();

        const VecVRef& index1 = this->f_index1.getValue();
        const VecVRef& index2 = this->f_index2.getValue();

        for( size_t i=0 ; i<this->maskTo[0]->size() ; ++i)
        {
            if( !this->maskTo[0]->isActivated() || this->maskTo[0]->getEntry(i) )
            {
                out[i]=OutDeriv();
                for(size_t j=0; j<jacobian1[i].size(); j++)
                {
                    size_t index=index1[i][j];
                    jacobian1[i][j].addmult(out[i],in1[index]);
                }
                for(size_t j=0; j<jacobian2[i].size(); j++)
                {
                    size_t index=index2[i][j];
                    jacobian2[i][j].addmult(out[i],in2[index]);
                }
            }
        }

        dOut.endEdit();
    }
}

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv1>& dIn1, Data<InVecDeriv2>& dIn2, const Data<OutVecDeriv>& dOut)
{
    if(this->assemble.getValue())
    {
        if( !eigenJacobian1.rows() ) updateJ1();
        if( !eigenJacobian2.rows() ) updateJ2();

//        if( this->maskTo[0]->isActivated() )
//        {
//            updateMaskedJ();
//            maskedEigenJacobian1.addMultTranspose(dIn1,dOut);
//            maskedEigenJacobian2.addMultTranspose(dIn2,dOut);
//        }
//        else
        {
            eigenJacobian1.addMultTranspose(dIn1,dOut);
            eigenJacobian2.addMultTranspose(dIn2,dOut);
        }
    }
    else
    {
        InVecDeriv1& in1 = *dIn1.beginEdit();
        InVecDeriv2& in2 = *dIn2.beginEdit();
        const OutVecDeriv& out = dOut.getValue();

        const VecVRef& index1 = this->f_index1.getValue();
        const VecVRef& index2 = this->f_index2.getValue();

        for( size_t i=0 ; i<this->maskTo[0]->size() ; ++i)
        {
            if( this->maskTo[0]->getEntry(i) )
            {
                for(size_t j=0; j<jacobian1[i].size(); j++)
                {
                    size_t index=index1[i][j];
                    jacobian1[i][j].addMultTranspose(in1[index],out[i]);
                }
                for(size_t j=0; j<jacobian2[i].size(); j++)
                {
                    size_t index=index2[i][j];
                    jacobian2[i][j].addMultTranspose(in2[index],out[i]);
                }
            }
        }

        dIn1.endEdit();
        dIn2.endEdit();
    }
}

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*parentDfId*/, core::ConstMultiVecDerivId )
{
    if( (BlockType1::constant && BlockType2::constant) || !d_geometricStiffness.getValue() ) return;

    serr<<"applyDJT is not implemented\n";
    return;

//    if(!this->isMechanical()) return;
//    if(BlockType1::constant && BlockType2::constant) return;

//    if(this->f_printLog.getValue()) sout<<":applyDJT"<<sendl;

//    const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);
//    helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);

//    // this does not compile for some reasons..
////    Data<InVecDeriv1>& parentForceData1 = *parentDfId[this->fromModel1.get(mparams)].write();
////    Data<InVecDeriv2>& parentForceData2 = *parentDfId[this->fromModel2.get(mparams)].write();
//    // work around..
//    helper::WriteAccessor<Data<InVecDeriv1> > parentForce1 (this->fromModel1->write(core::VecDerivId::force()));
//    helper::WriteAccessor<Data<InVecDeriv2> > parentForce2 (this->fromModel2->write(core::VecDerivId::force()));

//    const Data<InVecDeriv1>& parentDisplacementData1 = *mparams->readDx(this->fromModel1) ;
//    helper::ReadAccessor<Data<InVecDeriv1> > parentDisplacement1 (parentDisplacementData1);
//    const Data<InVecDeriv2>& parentDisplacementData2 = *mparams->readDx(this->fromModel2) ;
//    helper::ReadAccessor<Data<InVecDeriv2> > parentDisplacement2 (parentDisplacementData2);

//    if(this->assemble.getValue())
//    {
//        if(!BlockType1::constant)
//        {
//            K.addMult(parentForce1.wref(),parentDisplacementData1.getValue(),mparams->kFactor());
//        }
//    }
//    else
//    {
//        if( !this->maskTo || !this->maskTo->isInUse() )
//        {
//            if(!BlockType1::constant)
//            {
//#ifdef _OPENMP
//#pragma omp parallel for if (this->d_parallel.getValue())
//#endif
//                for(unsigned int i=0; i<this->f_index_parentToChild1.size(); i++)
//                {
//                    for(size_t j=0; j<this->f_index_parentToChild1[i].size(); j+=2)
//                    {
//                        size_t indexc=this->f_index_parentToChild1[i][j];
//                        jacobian1[indexc][this->f_index_parentToChild1[i][j+1]].addDForce(parentForce1[i],parentDisplacement1[i],childForce[indexc], mparams->kFactor());
//                    }
//                }
//            }
//            if(!BlockType2::constant)
//            {
//#ifdef _OPENMP
//#pragma omp parallel for if (this->d_parallel.getValue())
//#endif
//                for(unsigned int i=0; i<this->f_index_parentToChild2.size(); i++)
//                {
//                    for(size_t j=0; j<this->f_index_parentToChild2[i].size(); j+=2)
//                    {
//                        size_t indexc=this->f_index_parentToChild2[i][j];
//                        jacobian2[indexc][this->f_index_parentToChild2[i][j+1]].addDForce(parentForce2[i],parentDisplacement2[i],childForce[indexc], mparams->kFactor());
//                    }
//                }
//            }
//        }
//        else
//        {
//            const helper::StateMask::InternalStorage &indices=this->maskTo->getEntries();
//            for (helper::StateMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
//            {
//                const int i= ( int ) ( *it );
//                if(!BlockType1::constant)
//                {
//                    for(size_t j=0; j<jacobian1[i].size(); j++)
//                    {
//                        size_t index=this->f_index1.getValue()[i][j];
//                        jacobian1[i][j].addDForce( parentForce1[index], parentDisplacement1[index], childForce[i], mparams->kFactor() );
//                    }
//                }
//                if(!BlockType2::constant)
//                {
//                    for(size_t j=0; j<jacobian2[i].size(); j++)
//                    {
//                        size_t index=this->f_index2.getValue()[i][j];
//                        jacobian2[i][j].addDForce( parentForce2[index], parentDisplacement2[index], childForce[i], mparams->kFactor() );
//                    }
//                }
//            }
//        }
//    }
}



template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::applyJT( const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv1>& _out1, Data<InMatrixDeriv2>& _out2, const Data<OutMatrixDeriv>& _in )
{
    // TODO handle mask

    InMatrixDeriv1& out1 = *_out1.beginEdit();
    InMatrixDeriv2& out2 = *_out2.beginEdit();
    const OutMatrixDeriv& in = _in.getValue();

    const VecVRef& index1 = this->f_index1.getValue();
    const VecVRef& index2 = this->f_index2.getValue();

    typename OutMatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename OutMatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename OutMatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename OutMatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename InMatrixDeriv1::RowIterator o1 = out1.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                size_t indexIn = colIt.index();

                for(size_t j=0; j<jacobian1[indexIn].size(); j++)
                {
                    size_t indexOut = index1[indexIn][j];

                    InDeriv1 tmp;
                    jacobian1[indexIn][j].addMultTranspose( tmp, colIt.val() );

                    o1.addCol( indexOut, tmp );
                }
            }
            typename InMatrixDeriv2::RowIterator o2 = out2.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                size_t indexIn = colIt.index();

                for(size_t j=0; j<jacobian2[indexIn].size(); j++)
                {
                    size_t indexOut = index2[indexIn][j];

                    InDeriv2 tmp;
                    jacobian2[indexIn][j].addMultTranspose( tmp, colIt.val() );

                    o2.addCol( indexOut, tmp );
                }
            }
        }
    }

    _out1.endEdit();
    _out2.endEdit();
}




/** abstract implementation of BasePointMapper functions
    they call mapPosition/mapDefoGradient specialized functions (templated on specific jacobianBlockType)
**/

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::ForwardMapping(Coord& p,const Coord& p0)
{
    if ( !_shapeFunction ) return;

    // interpolate weights at sample positions
    mCoord mp0;        defaulttype::StdVectorTypes<mCoord,mCoord>::set( mp0, p0[0] , p0[1] , p0[2]);
    VRef ref; VReal w;
    _shapeFunction->computeShapeFunction(mp0,ref,w);

    // map using specific instanciation
    this->mapPosition(p,p0,ref,w);
}


template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::BackwardMapping(Coord& p0,const Coord& p,const Real Thresh, const size_t NbMaxIt)
{
    if ( !_shapeFunction ) return;

    // iterate: p0(n+1) = F0.F^-1 (p-p(n)) + p0(n)
    size_t count=0;
    mCoord mp0;
    MaterialToSpatial F0;  VRef ref; VReal w; VGradient dw;
    Coord pnew;
    MaterialToSpatial F;
    defaulttype::Mat<material_dimensions,spatial_dimensions,Real> Finv;

    identity(F0);
    while(count<NbMaxIt)
    {
        defaulttype::StdVectorTypes<mCoord,mCoord>::set( mp0, p0[0] , p0[1] , p0[2]);
        _shapeFunction->computeShapeFunction(mp0,ref,w,&dw);
        if(!w[0]) { p0=Coord(); return; } // outside object

        this->mapPosition(pnew,p0,ref,w);
        if((p-pnew).norm2()<Thresh) return; // has converged
        this->mapDeformationGradient(F,p0,F0,ref,w,dw);

        invert(Finv,F);
        p0+=F0*Finv*(p-pnew);
        count++;
    }
}


template <class JacobianBlockType1,class JacobianBlockType2>
unsigned int BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree)
{
    helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    if(this->missingInformationDirty)
    {
        if(!OutDataTypesInfo<Out>::positionMapped) mapPositions();
        this->missingInformationDirty=false;
    }

    size_t index=0;
    if(useKdTree)
    {
        if(this->KdTreeDirty)
        {
            if(OutDataTypesInfo<Out>::positionMapped) { f_pos.resize(out.size()); for(size_t i=0; i<out.size(); i++ )  Out::get(f_pos[i][0],f_pos[i][1],f_pos[i][2],out[i]);  } // copy to f_pos
            this->f_KdTree.build(f_pos);
            this->KdTreeDirty=false;
        }
        index=this->f_KdTree.getClosest(p,f_pos);
        x=f_pos[index];
    }
    else
    {
        Real dmin=std::numeric_limits<Real>::max();
        for(size_t i=0; i<out.size(); i++)
        {
            Coord P; if(OutDataTypesInfo<Out>::positionMapped) Out::get(P[0],P[1],P[2],out[i]); else P=f_pos[i];
            Real d=(p-P).norm2();
            if(d<dmin) {dmin=d; index=i; x=P;}
        }
    }
    x0=f_pos0.getValue()[index];
    return index;
}


template <class JacobianBlockType1,class JacobianBlockType2>
const helper::vector<sofa::defaulttype::BaseMatrix*>* BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::getJs()
{
    if( !this->assemble.getValue() )
    {
        updateJ1();
        updateJ2();
    }

    if( !eigenJacobian1.rows() ) updateJ1();
    if( !eigenJacobian2.rows() ) updateJ2();

//    if( this->maskTo[0]->isActivated() )
//    {
//        updateMaskedJ();
//        baseMatrices[0] = &maskedEigenJacobian1;
//        baseMatrices[1] = &maskedEigenJacobian2;
//    }
//    else
//    {
//        baseMatrices[0] = &eigenJacobian1;
//        baseMatrices[1] = &eigenJacobian2;
//    }

    return &baseMatrices;
}




template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowMechanicalMappings() && !showDeformationGradientScale.getValue() && showColorOnTopology.getValue().getSelectedId()==0) return;


    glPushAttrib ( GL_LIGHTING_BIT );

    helper::ReadAccessor<Data<InVecCoord1> > in1 (*this->fromModel1->read(core::ConstVecCoordId::position()));
    helper::ReadAccessor<Data<InVecCoord2> > in2 (*this->fromModel2->read(core::ConstVecCoordId::position()));
    helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    helper::ReadAccessor<Data<VecVRef > > ref (this->f_index);
    helper::ReadAccessor<Data<VecVReal > > w (this->f_w);
    size_t size1=this->getFromSize1();

    if(this->missingInformationDirty)
    {
        if(!OutDataTypesInfo<Out>::positionMapped) mapPositions();
        if(!OutDataTypesInfo<Out>::FMapped) if(showDeformationGradientScale.getValue() || showColorOnTopology.getValue().getSelectedId()!=0) mapDeformationGradients();
        this->missingInformationDirty=false;
    }

    if (vparams->displayFlags().getShowMechanicalMappings())
    {
        helper::vector< defaulttype::Vector3 > edge;     edge.resize(2);
        defaulttype::Vec<4,float> col;

        for(size_t i=0; i<out.size(); i++ )
        {
            if(OutDataTypesInfo<Out>::positionMapped) Out::get(edge[1][0],edge[1][1],edge[1][2],out[i]);
            else edge[1]=f_pos[i];
            for(size_t j=0; j<ref[i].size(); j++ )
                if(w[i][j])
                {
                    if(j<size1) In1::get(edge[0][0],edge[0][1],edge[0][2],in1[ref[i][j]]);
                    else In2::get(edge[0][0],edge[0][1],edge[0][2],in2[ref[i][j]-size1]);
                    sofa::helper::gl::Color::getHSVA(&col[0],240.*w[i][j],1.,.8,1.);
                    vparams->drawTool()->drawLines ( edge, 1, col );
                }
        }
    }
    if (showDeformationGradientScale.getValue())
    {
        const Data<OutVecDeriv>* outf = this->toModel->read(core::ConstVecDerivId::force());
        glEnable ( GL_LIGHTING );
        float scale=showDeformationGradientScale.getValue();
        defaulttype::Vec<4,float> col( 0.5, 0.5, 0.0, 1.0 );
        defaulttype::Mat<3,3,float> F;
        defaulttype::Vec<3,float> p;

        static const int subdiv = 8;

        for(size_t i=0; i<out.size(); i++ )
        {
            if(OutDataTypesInfo<Out>::FMapped) F=(defaulttype::Mat<3,3,float>)OutDataTypesInfo<Out>::getF(out[i]); else F=(defaulttype::Mat<3,3,float>)f_F[i];
            if(OutDataTypesInfo<Out>::positionMapped) Out::get(p[0],p[1],p[2],out[i]); else p=f_pos[i];

            if(showDeformationGradientStyle.getValue().getSelectedId()==0)
                for(int j=0; j<material_dimensions; j++)
                {
                    defaulttype::Vec<3,float> u=F.transposed()(j)*0.5*scale;
                    vparams->drawTool()->drawCylinder(p-u,p+u,0.05*scale,col,subdiv);
                }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==1)
            {
                defaulttype::Vec<3,float> u=F.transposed()(0)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05*scale,col,subdiv);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==2)
            {
                defaulttype::Vec<3,float> u=F.transposed()(1)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05*scale,col,subdiv);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==3)
            {
                defaulttype::Vec<3,float> u=F.transposed()(2)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05*scale,col,subdiv);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==4) // strain
            {
                vparams->drawTool()->setMaterial(col);
                drawEllipsoid(F,p,0.5*scale);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==5 && outf) // stress
                if(OutDataTypesInfo<Out>::FMapped)
                {
                    F=(defaulttype::Mat<3,3,float>)OutDataTypesInfo<Out>::getF(outf->getValue()[i]);
                    vparams->drawTool()->setMaterial(col);
                    drawEllipsoid(F,p,0.5*scale);
                }

        }
    }

    if(showColorOnTopology.getValue().getSelectedId() && (this->extTriangles || this->triangles))
    {
        std::vector< Real > val(out.size());
        MaterialToSpatial F;
        for(size_t i=0; i<out.size(); i++ )
        {
            if(OutDataTypesInfo<Out>::FMapped) F=OutDataTypesInfo<Out>::getF(out[i]); else F=f_F[i];

            if(showColorOnTopology.getValue().getSelectedId()==1) val[i]=(defaulttype::trace(F.transposed()*F)-3.);
            else  val[i]=sqrt(defaulttype::determinant(F.transposed()*F))-1.;

            //if (val[i]<0) val[i]=2*val[i]/(val[i]+1.);
            val[i]*=240 * this->showColorScale.getValue();
            val[i]+=120;
            if (val[i]<0) val[i]=0;
            if (val[i]>240) val[i]=240;
        }

        size_t nb =0;
        if(triangles) nb+=triangles->size();
        if(extTriangles) nb+=extTriangles->size();

        std::vector< defaulttype::Vector3 > points(3*nb),normals;
        std::vector< defaulttype::Vec<4,float> > colors(3*nb);
        size_t count=0;

        if(triangles)
            for ( size_t i = 0; i < triangles->size(); i++)
                for ( size_t j = 0; j < 3; j++)
                {
                    size_t index = (*triangles)[i][j];
                    if(OutDataTypesInfo<Out>::positionMapped) Out::get(points[count][0],points[count][1],points[count][2],out[index]); else points[count]=f_pos[index];
                    sofa::helper::gl::Color::getHSVA(&colors[count][0],val[index],1.,.8,1.);
                    count++;
                }
        if(extTriangles)
            for ( size_t i = 0; i < extTriangles->size(); i++)
                for ( size_t j = 0; j < 3; j++)
                {
                    size_t index = (*extTriangles)[i][j];
                    if(this->extvertPosIdx) index=(*extvertPosIdx)[index];
                    if(OutDataTypesInfo<Out>::positionMapped) Out::get(points[count][0],points[count][1],points[count][2],out[index]); else points[count]=f_pos[index];
                    sofa::helper::gl::Color::getHSVA(&colors[count][0],val[index],1.,.8,1.);
                    count++;
                }

        glDisable( GL_LIGHTING);
        vparams->drawTool()->drawTriangles(points, normals, colors);
    }
    glPopAttrib();
#endif /* SOFA_NO_OPENGL */
}

template <class JacobianBlockType1,class JacobianBlockType2>
void BaseDeformationMultiMappingT<JacobianBlockType1,JacobianBlockType2>::updateForceMask()
{
    const VecVRef& indices1 = this->f_index1.getValue();
    const VecVRef& indices2 = this->f_index2.getValue();

    for( size_t i=0 ; i<this->maskTo[0]->size() ; ++i)
    {
        if( this->maskTo[0]->getEntry(i) )
        {
            for(size_t j=0; j<jacobian1[i].size(); j++)
            {
                size_t index = indices1[i][j];
                this->maskFrom1[0]->insertEntry( index );
            }
            for(size_t j=0; j<jacobian2[i].size(); j++)
            {
                size_t index = indices2[i][j];
                this->maskFrom2[0]->insertEntry( index );
            }
        }
    }
}

} // namespace mapping
} // namespace component
} // namespace sofa

#endif

