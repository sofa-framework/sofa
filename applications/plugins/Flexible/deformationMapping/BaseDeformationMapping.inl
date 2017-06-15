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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_INL

#include "BaseDeformationMapping.h"
#include "BaseDeformationImpl.inl"
#include <SofaBaseVisual/VisualModelImpl.h>
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

template <class JacobianBlockType>
BaseDeformationMappingT<JacobianBlockType>::BaseDeformationMappingT (core::State<In>* from , core::State<Out>* to)
    : Inherit ( from, to )
    , f_shapeFunction_name(initData(&f_shapeFunction_name,"shapeFunction","name of shape function (optional)"))
    , _shapeFunction(NULL)
    , _sampler(NULL)
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , f_w ( initData ( &f_w,"weights","influence weights of the Dofs" ) )
    , f_dw ( initData ( &f_dw,"weightGradients","weight gradients" ) )
    , f_ddw ( initData ( &f_ddw,"weightHessians","weight Hessians" ) )
    , f_F0 ( initData ( &f_F0,"M","Linear transformations from material to 3d space" ) )
    , f_cell ( initData ( &f_cell,"cell","indices required by shape function in case of overlapping elements" ) )
    , assemble ( initData ( &assemble,false, "assemble","Assemble the matrices (Jacobian/Geometric Stiffness) or use optimized Jacobian/vector multiplications" ) )
    , f_pos0 ( initData ( &f_pos0,"restPosition","initial spatial positions of children" ) )
    , missingInformationDirty(true)
    , KdTreeDirty(true)
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

/*
template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::updateIndex(const size_t parentSize, const size_t childSize)
{
    if(this->f_printLog.getValue())
        std::cout<<this->getName()<< "::" << SOFA_CLASS_METHOD <<std::endl;

    this->f_index_parentToChild.clear();
    this->f_index_parentToChild.resize(parentSize);

    const VecVRef& index = this->f_index.getValue();

    //Check size just in case
    if(childSize != index.size())
    {
        std::cout << SOFA_CLASS_METHOD << " f_index has wrong size" << std::endl;
        serr << "index size : " << index.size() << sendl;
        serr << "child size : " << childSize << sendl;
        exit(EXIT_FAILURE);
    }

    //Go through f_index and use its value to update f_index_parentToChild
    for(size_t i=0; i<index.size(); ++i)
    {
        for(size_t j=0; j< index[i].size(); j++ )
        {
            int parentIndex = index[i][j];
            this->f_index_parentToChild[parentIndex].push_back(i); //Add child index
            this->f_index_parentToChild[parentIndex].push_back(j); //Add parent index
        }
    }
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::updateIndex()
{    
    if(this->f_printLog.getValue())
    {
        std::cout<<this->getName()<< "::" << SOFA_CLASS_METHOD <<std::endl;
    }

    int parentSize = this->getFromSize();
    int childSize = this->getToSize();
    this->f_index_parentToChild.clear();
    this->f_index_parentToChild.resize(parentSize);

    if(this->f_printLog.getValue())
    {
        std::cout << "parent size : " << parentSize << std::endl;
        std::cout << "child size : " << childSize << std::endl;
    }


    const VecVRef& index = this->f_index.getValue();

    //Check size just in case
    if( (unsigned)childSize != index.size() )
    {
        std::cout << SOFA_CLASS_METHOD << " f_index has wrong size" << std::endl;
        serr << "index size : " << index.size() << sendl;
        serr << "child size : " << childSize << sendl;
        exit(EXIT_FAILURE);
    }

    //Go through f_index and use its value to update f_index_parentToChild
    for(size_t i=0; i<index.size(); ++i)
    {
        for(size_t j=0; j< index[i].size(); j++ )
        {
            int parentIndex = index[i][j];
            this->f_index_parentToChild[parentIndex].push_back(i); //Add child index
            this->f_index_parentToChild[parentIndex].push_back(j); //Add parent index
        }
    }
}
*/

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::resizeAll(const InVecCoord& p0, const OutVecCoord& c0, const VecCoord& x0, const VecVRef& index, const VecVReal& w, const VecVGradient& dw, const VecVHessian& ddw, const VMaterialToSpatial& F0)
{
    size_t cSize = c0.size();
    if(cSize != x0.size() || cSize != index.size() || cSize != w.size() || cSize != dw.size() || cSize != ddw.size())
    {
        sout << SOFA_CLASS_METHOD << " : wrong sizes " << sendl;
    }

    helper::WriteOnlyAccessor<Data<VecCoord> > wa_x0 (this->f_pos0);
    wa_x0.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_x0[i] = x0[i];

    helper::WriteOnlyAccessor<Data<VecVRef > > wa_index (this->f_index);
    wa_index.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_index[i].assign(index[i].begin(), index[i].end());

    helper::WriteOnlyAccessor<Data<VecVReal > > wa_w (this->f_w);
    wa_w.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_w[i].assign(w[i].begin(), w[i].end());

    helper::WriteOnlyAccessor<Data<VecVGradient > > wa_dw (this->f_dw);
    wa_dw.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_dw[i].assign(dw[i].begin(), dw[i].end());

    helper::WriteOnlyAccessor<Data<VecVHessian > > wa_ddw (this->f_ddw);
    wa_ddw.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_ddw[i].assign(ddw[i].begin(), ddw[i].end());

    helper::WriteOnlyAccessor<Data< VMaterialToSpatial > >  wa_F0(this->f_F0);
    wa_F0.resize(cSize);
    for(size_t i=0; i<cSize; ++i)
        wa_F0[i] = F0[i];

//    updateIndex(p0.size(), c0.size());

    initJacobianBlocks(p0, c0);
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::resizeOut()
{
    {
        // TODO this must be done before resizeOut() but is done again in Inherit::init();
        // also clean the numerous calls to apply
        core::behavior::BaseMechanicalState *state;
        if ((state = this->fromModel.get()->toBaseMechanicalState()))
            this->maskFrom = &state->forceMask;
        if ((state = this->toModel.get()->toBaseMechanicalState()))
            this->maskTo = &state->forceMask;
        else
            this->setNonMechanical();
    }

    helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

    helper::WriteOnlyAccessor<Data<VecCoord> > pos0 (this->f_pos0);
    helper::WriteOnlyAccessor<Data< VMaterialToSpatial > > F0(this->f_F0);
    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update mapped spatial positions if needed for visualization

    size_t size;

    if( !_sampler ) this->getContext()->get(_sampler,core::objectmodel::BaseContext::Local);
    bool restPositionSet=false;
    helper::ReadAccessor<Data< OutVecCoord > >  rest(*this->toModel->read(core::ConstVecCoordId::restPosition()));

    if(_sampler) // retrieve initial positions from gauss point sampler (deformation gradient types)
    {
        size = _sampler->getNbSamples();
        if(rest.size()==size && size!=1)  restPositionSet=true;

        this->toModel->resize(size);
        pos0.resize(size);  for(size_t i=0; i<size; i++) pos0[i]=_sampler->getSamples()[i];
        F0.resize(size);
        if(restPositionSet)     // use custom rest positions defined in state (to set material directions or set residual deformations)
        {
            for(size_t i=0; i<rest.size(); ++i) F0[i]=OutDataTypesInfo<Out>::getF(rest[i]);
            sout<<rest.size()<<" rest positions imported "<<sendl;
        }
        else
        {
            for(size_t i=0; i<size; ++i) copy(F0[i],_sampler->getTransforms()[i]);
        }
        sout<<size <<" gauss points imported"<<sendl;
    }
    else  // retrieve initial positions from children dofs (vec types)
    {
        size = out.size();
        this->toModel->resize(size);
        pos0.resize(size);  for(size_t i=0; i<size; i++ )  Out::get(pos0[i][0],pos0[i][1],pos0[i][2],out[i]);
        //        F0.resize(size); for(size_t i=0; i<size; ++i) identity(F0[i]); // necessary ?
    }

    // init shape function
    if( !_shapeFunction )
    {
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        std::vector<BaseShapeFunction*> sf; context->get<BaseShapeFunction>(&sf,core::objectmodel::BaseContext::SearchUp);
        for(unsigned int i=0;i<sf.size();i++)
        {
            if(this->f_shapeFunction_name.isSet()) {if(this->f_shapeFunction_name.getValue().compare(sf[i]->getName()) == 0) _shapeFunction=sf[i];}
            else if(sf[i]->f_position.getValue().size() == this->fromModel->getSize()) _shapeFunction=sf[i];
        }
    }

    if(0 != f_index.getValue().size() && pos0.size() == f_index.getValue().size() && f_w.getValue().size() == f_index.getValue().size()) // we already have the needed data, we directly use them
    {
        sout<<"using filled data" <<sendl;
    }
    else if(_shapeFunction) // if we do not have the needed data, and have a shape function, we use it to compute needed data (index, weights, etc.)
    {
        sout<<"found shape function "<<_shapeFunction->getName()<<sendl;
        helper::vector<mCoord> mpos0;
        mpos0.resize(pos0.size());
        for(size_t i=0; i<pos0.size(); ++i) defaulttype::StdVectorTypes<mCoord,mCoord>::set( mpos0[i], pos0[i][0] , pos0[i][1] , pos0[i][2]);

        // interpolate weights at sample positions
        if(this->f_cell.getValue().size()==size) _shapeFunction->computeShapeFunction(mpos0,*this->f_index.beginWriteOnly(),*this->f_w.beginWriteOnly(),*this->f_dw.beginWriteOnly(),*this->f_ddw.beginWriteOnly(),this->f_cell.getValue());
        else _shapeFunction->computeShapeFunction(mpos0,*this->f_index.beginWriteOnly(),*this->f_w.beginWriteOnly(),*this->f_dw.beginWriteOnly(),*this->f_ddw.beginWriteOnly());
        this->f_index.endEdit();      this->f_w.endEdit();        this->f_dw.endEdit();        this->f_ddw.endEdit();
    }
    else // if the prerequisites are not fulfilled we print an error
    {
        serr << "ShapeFunction<"<<ShapeFunctionType::Name()<<"> component not found" << sendl;
    }

//    updateIndex();

    // init jacobians
    initJacobianBlocks();

    // clear forces
    if(this->toModel->write(core::VecDerivId::force())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  f(*this->toModel->write(core::VecDerivId::force())); for(size_t i=0;i<f.size();i++) f[i].clear(); }
    // clear velocities
    if(this->toModel->write(core::VecDerivId::velocity())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  vel(*this->toModel->write(core::VecDerivId::velocity())); for(size_t i=0;i<vel.size();i++) vel[i].clear(); }

    //Apply mapping to init child positions
    reinit();

    // set deformation gradient state rest position when defined by gaussPointSampler
    if(_sampler && restPositionSet == false && this->toModel->read(core::VecCoordId::restPosition())->getValue().size()==size ) // not for states that do not have restpos (like visualmodel)
    {
        helper::ReadAccessor<Data< VMaterialToSpatial > >  ra_F0(this->f_F0);
        helper::WriteOnlyAccessor<Data< OutVecCoord > >  rest(*this->toModel->write(core::VecCoordId::restPosition()));
        for(size_t i=0; i<rest.size(); ++i) for(int j=0; j<spatial_dimensions; ++j) for(int k=0; k<material_dimensions; ++k) rest[i][j*material_dimensions+k] = (OutReal)ra_F0[i][j][k];
    }
}



template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::resizeOut(const helper::vector<Coord>& position0, helper::vector<helper::vector<unsigned int> > index,helper::vector<helper::vector<Real> > w, helper::vector<helper::vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, helper::vector<helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0)
{
    {
        // TODO this must be done before resizeOut() but is done again in Inherit::init();
        // also clean the numerous calls to apply
        core::behavior::BaseMechanicalState *state;
        if( (state = this->fromModel.get()->toBaseMechanicalState()) )
            this->maskFrom = &state->forceMask;
        if( (state = this->toModel.get()->toBaseMechanicalState()) )
            this->maskTo = &state->forceMask;
        else
            this->setNonMechanical();
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

//    updateIndex();

    // init jacobians
    initJacobianBlocks();

    // clear forces
    if(this->toModel->write(core::VecDerivId::force())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  f(*this->toModel->write(core::VecDerivId::force())); for(size_t i=0;i<f.size();i++) f[i].clear(); }
    // clear velocities
    if(this->toModel->write(core::VecDerivId::velocity())) { helper::WriteOnlyAccessor<Data< OutVecDeriv > >  vel(*this->toModel->write(core::VecDerivId::velocity())); for(size_t i=0;i<vel.size();i++) vel[i].clear(); }

    //Apply mapping to init child positions
    reinit();

    // update deformation gradient state rest position
    if( this->toModel->read(core::VecCoordId::restPosition())->getValue().size()==size ) // not for states that do not have restpos (like visualmodel)
    {
        if( !_sampler ) this->getContext()->get(_sampler,core::objectmodel::BaseContext::Local);
        if(_sampler )
        {
            helper::ReadAccessor<Data< VMaterialToSpatial > >  ra_F0(this->f_F0);
            helper::WriteOnlyAccessor<Data< OutVecCoord > >  rest(*this->toModel->write(core::VecCoordId::restPosition()));
            for(size_t i=0; i<rest.size(); ++i) for(int j=0; j<spatial_dimensions; ++j) for(int k=0; k<material_dimensions; ++k) rest[i][j*material_dimensions+k] = (OutReal)ra_F0[i][j][k];
        }
    }
}


template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::init()
{
    baseMatrices.resize( 1 ); // just a wrapping for getJs()
    baseMatrices[0] = &eigenJacobian;

    resizeOut();

    Inherit::init();

    // check that all children particles got a parent
    const VecVRef& indices = this->f_index.getValue();
    for (std::size_t i=0; i < indices.size(); ++i)
        if ( indices[i].empty() )
            serr << this->getPathName() << " Particle " << i << " has no parent" << sendl;
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::reinit()
{
    if(this->isMechanical() && this->assemble.getValue()) updateJ();

    // force apply
    // bg: do we need this ?
    //    apply(NULL, *this->toModel->write(core::VecCoordId::position()), *this->fromModel->read(core::ConstVecCoordId::position()));
    //    if(this->toModel->write(core::VecDerivId::velocity())) applyJ(NULL, *this->toModel->write(core::VecDerivId::velocity()), *this->fromModel->read(core::ConstVecDerivId::velocity()));

    Inherit::reinit();
}



template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::updateJ()
{
    helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    //helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    const VecVRef& index = this->f_index.getValue();

    SparseMatrixEigen& J = eigenJacobian;

    J.resizeBlocks(jacobian.size(),in.size());

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        J.beginBlockRow(i);
        for(size_t j=0; j<jacobian[i].size(); j++)
            J.createBlock( index[i][j], jacobian[i][j].getJ());
        J.endBlockRow();
    }

    J.compress();

    //    maskedEigenJacobian.resize(0,0);
}


//template <class JacobianBlockType>
//void BaseDeformationMappingT<JacobianBlockType>::updateMaskedJ()
//{
//    size_t currentHash = this->maskTo->getHash();
//    if( previousMaskHash!=currentHash )
//    {
//        previousMaskHash = currentHash;
//        maskedEigenJacobian.resize(0,0);
//    }
//    if( !maskedEigenJacobian.rows() )
//    {
//        this->maskTo->maskedMatrix( maskedEigenJacobian.compressedMatrix, eigenJacobian.compressedMatrix, Out::deriv_total_size );
//        sout<<"updateMaskedJ "<<maskedEigenJacobian.compressedMatrix.nonZeros()<<sendl;
//    }
//}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId )
{
    unsigned geometricStiffness = d_geometricStiffness.getValue();

    if( BlockType::constant || !geometricStiffness /*|| !assemble.getValue()*/ ) { K.resize(0,0); return; }

    const OutVecDeriv& childForce = childForceId[this->toModel.get(mparams)].read()->getValue();
    helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    const VecVRef& indices = this->f_index.getValue();

    K.resizeBlocks(in.size(),in.size());
    helper::vector<KBlock> diagonalBlocks; diagonalBlocks.resize(in.size());

    // TODO: need to take into account mask in geometric stiffness, I do no think so!??

    for(size_t i=0; i<jacobian.size(); i++)
    {
        for(size_t j=0; j<jacobian[i].size(); j++)
        {
            size_t index=indices[i][j];
            diagonalBlocks[index] += jacobian[i][j].getK(childForce[i], geometricStiffness==2);
        }
    }

    for(size_t i=0; i<in.size(); i++)
        K.insertBackBlock(i,i,diagonalBlocks[i]);
    K.compress();
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::apply(OutVecCoord& out, const InVecCoord& in)
{
    const VecVRef& indices = this->f_index.getValue();

#ifdef _OPENMP
#pragma omp parallel for if (this->d_parallel.getValue())
#endif
    for(helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian.size(); i++)
    {
        out[i]=OutCoord();
        for(size_t j=0; j<jacobian[i].size(); j++)
        {
            size_t index=indices[i][j];
            jacobian[i][j].addapply(out[i],in[index]);
        }
    }

    if(this->assemble.getValue() && ( !BlockType::constant ) )  eigenJacobian.resize(0,0); // J needs to be updated later where the dof mask can be activated

    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update spatial positions of defo grads if needed for visualization
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::applyJ(OutVecDeriv& out, const InVecDeriv& in)
{
    if(this->assemble.getValue())
    {
        if( !eigenJacobian.rows() ) updateJ();
        eigenJacobian.mult(out,in);
    }
    else
    {
        const VecVRef& indices = this->f_index.getValue();

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( !this->maskTo->isActivated() || this->maskTo->getEntry(i) )
            {
                out[i]=OutDeriv();
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addmult(out[i],in[index]);
                }
            }
        }
    }
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    OutVecCoord& out = *dOut.beginWriteOnly();
    const InVecCoord& in = dIn.getValue();
    const VecVRef& indices = this->f_index.getValue();

#ifdef _OPENMP
#pragma omp parallel for if (this->d_parallel.getValue())
#endif
    for(helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian.size(); i++)
    {
        out[i]=OutCoord();
        for(size_t j=0; j<jacobian[i].size(); j++)
        {
            size_t index=indices[i][j];
            jacobian[i][j].addapply(out[i],in[index]);
        }
    }
    dOut.endEdit();

    if(this->assemble.getValue() && ( !BlockType::constant ) ) eigenJacobian.resize(0,0); // J needs to be updated later where the dof mask can be activated

    this->missingInformationDirty=true; this->KdTreeDirty=true; // need to update spatial positions of defo grads if needed for visualization
}



template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if(this->assemble.getValue())
    {
        if( !eigenJacobian.rows() ) updateJ();

        //        if( this->maskTo->isActivated() )
        //        {
        //            updateMaskedJ();
        //            maskedEigenJacobian.mult(dOut,dIn);
        //        }
        //        else
        eigenJacobian.mult(dOut,dIn);
    }
    else
    {
        OutVecDeriv& out = *dOut.beginWriteOnly();
        const InVecDeriv& in = dIn.getValue();
        const VecVRef& indices = this->f_index.getValue();

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( !this->maskTo->isActivated() || this->maskTo->getEntry(i) )
            {
                out[i]=OutDeriv();
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addmult(out[i],in[index]);
                }
            }
        }

        dOut.endEdit();
    }
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if(this->assemble.getValue())
    {
        if( !eigenJacobian.rows() ) updateJ();

        //        if( this->maskTo->isActivated() )
        //        {
        //            updateMaskedJ();
        //            maskedEigenJacobian.addMultTranspose(dIn,dOut);
        //        }
        //        else
        eigenJacobian.addMultTranspose(dIn,dOut);
    }
    else
    {
        InVecDeriv& in = *dIn.beginEdit();
        const OutVecDeriv& out = dOut.getValue();
        const VecVRef& indices = this->f_index.getValue();

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( this->maskTo->getEntry(i) )
            {
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addMultTranspose(in[index],out[i]);
                }
            }
        }

        dIn.endEdit();
    }
}

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId childForceId )
{
    if( BlockType::constant || !d_geometricStiffness.getValue() ) return;

    Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
    const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
    const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);

    helper::WriteAccessor<Data<InVecDeriv> > parentForce (parentForceData);
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (parentDisplacementData);
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);

    if( assemble.getValue() && K.compressedMatrix.nonZeros() ) // assembled version
    {
        assert( this->assemble.getValue() );
        K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
    }
    else
    {
        // if symmetrized version, force local assembly
        if( assemble.getValue() || d_geometricStiffness.getValue() == 2 )
        {
            updateK( mparams, childForceId );
            K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
            K.resize(0,0); // forgot about this matrix
        }
        else
        {

        const VecVRef& indices = this->f_index.getValue();
        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( this->maskTo->getEntry(i) )
            {
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addDForce(parentForce[index],parentDisplacement[index],childForce[i], mparams->kFactor());
                }
            }
        }

//#ifdef _OPENMP
//#pragma omp parallel for if (this->d_parallel.getValue())
//#endif
//            for(helper::IndexOpenMP<unsigned int>::type i=0; i<this->f_index_parentToChild.size(); i++)
//            {
//                for(size_t j=0; j<this->f_index_parentToChild[i].size(); j+=2)
//                {
//                    size_t indexc=this->f_index_parentToChild[i][j];
//                    jacobian[indexc][this->f_index_parentToChild[i][j+1]].addDForce(parentForce[i],parentDisplacement[i],childForce[indexc], mparams->kFactor());
//                }
//            }

        }
    }
}



template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::applyJT( const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& _out, const Data<OutMatrixDeriv>& _in )
{
    // TODO handle mask

    InMatrixDeriv& out = *_out.beginEdit();
    const OutMatrixDeriv& in = _in.getValue();
    const VecVRef& indices = this->f_index.getValue();

    typename OutMatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename OutMatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename OutMatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename OutMatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                size_t indexIn = colIt.index();

                for(size_t j=0; j<jacobian[indexIn].size(); j++)
                {
                    size_t indexOut = indices[indexIn][j];

                    InDeriv tmp;
                    jacobian[indexIn][j].addMultTranspose( tmp, colIt.val() );

                    o.addCol( indexOut, tmp );
                }
            }
        }
    }

    _out.endEdit();
}




/** abstract implementation of BasePointMapper functions
    they call mapPosition/mapDefoGradient specialized functions (templated on specific jacobianBlockType)
**/

template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::ForwardMapping(Coord& p,const Coord& p0)
{
    if ( !_shapeFunction ) return;

    // interpolate weights at sample positions
    mCoord mp0;        defaulttype::StdVectorTypes<mCoord,mCoord>::set( mp0, p0[0] , p0[1] , p0[2]);
    VRef ref; VReal w;
    _shapeFunction->computeShapeFunction(mp0,ref,w);

    // map using specific instanciation
    this->mapPosition(p,p0,ref,w);
}


template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::BackwardMapping(Coord& p0,const Coord& p,const Real Thresh, const size_t NbMaxIt)
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


template <class JacobianBlockType>
unsigned int BaseDeformationMappingT<JacobianBlockType>::getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree)
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


template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowMechanicalMappings() && !showDeformationGradientScale.getValue() && showColorOnTopology.getValue().getSelectedId()==0) return;


    glPushAttrib ( GL_LIGHTING_BIT );

    helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    helper::ReadAccessor<Data<VecVRef > > ref (this->f_index);
    helper::ReadAccessor<Data<VecVReal > > w (this->f_w);

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
                    In::get(edge[0][0],edge[0][1],edge[0][2],in[ref[i][j]]);
                    sofa::helper::gl::Color::getHSVA(&col[0],240.f*(float)w[i][j],1.f,.8f,1.f);
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
                    vparams->drawTool()->drawCylinder(p-u,p+u,0.05f*scale,col,subdiv);
                }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==1)
            {
                defaulttype::Vec<3,float> u=F.transposed()(0)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05f*scale,col,subdiv);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==2)
            {
                defaulttype::Vec<3,float> u=F.transposed()(1)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05f*scale,col,subdiv);
            }
            else if(showDeformationGradientStyle.getValue().getSelectedId()==3)
            {
                defaulttype::Vec<3,float> u=F.transposed()(2)*0.5*scale;
                vparams->drawTool()->drawCylinder(p-u,p+u,0.05f*scale,col,subdiv);
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
                    drawEllipsoid(F,p,0.5f*scale);
                }

        }
    }

    if(showColorOnTopology.getValue().getSelectedId() )
    {
        if( !this->extTriangles && !this->triangles )
        {
            component::visualmodel::VisualModelImpl *visual = NULL;
            this->getContext()->get( visual, core::objectmodel::BaseContext::Local);
            if(visual) {this->extTriangles = &visual->getTriangles(); this->extvertPosIdx = &visual->m_vertPosIdx.getValue(); this->triangles=0; }
            else
            {
                core::topology::BaseMeshTopology *topo = NULL;
                this->getContext()->get( topo, core::objectmodel::BaseContext::Local);
                if(topo) {this->triangles = &topo->getTriangles();  this->extTriangles=0; }
            }
        }

        if( this->extTriangles || this->triangles )
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
                        sofa::helper::gl::Color::getHSVA(&colors[count][0],(float)val[index],1.f,.8f,1.f);
                        count++;
                    }
            if(extTriangles)
                for ( size_t i = 0; i < extTriangles->size(); i++)
                    for ( size_t j = 0; j < 3; j++)
                    {
                        size_t index = (*extTriangles)[i][j];
                        if(this->extvertPosIdx) index=(*extvertPosIdx)[index];
                        if(OutDataTypesInfo<Out>::positionMapped) Out::get(points[count][0],points[count][1],points[count][2],out[index]); else points[count]=f_pos[index];
                        sofa::helper::gl::Color::getHSVA(&colors[count][0],(float)val[index],1.f,.8f,1.f);
                        count++;
                    }

            glDisable( GL_LIGHTING);
            vparams->drawTool()->drawTriangles(points, normals, colors);
        }
    }
    glPopAttrib();
#endif /* SOFA_NO_OPENGL */
}


template <class JacobianBlockType>
const defaulttype::BaseMatrix* BaseDeformationMappingT<JacobianBlockType>::getJ(const core::MechanicalParams * /*mparams*/)
{
    if(!this->assemble.getValue() || !BlockType::constant || !eigenJacobian.rows()) updateJ();

    //    if( this->maskTo->isActivated() )
    //    {
    //        updateMaskedJ();
    //        return &maskedEigenJacobian;
    //    }

    return &eigenJacobian;
}


template <class JacobianBlockType>
const helper::vector<sofa::defaulttype::BaseMatrix*>* BaseDeformationMappingT<JacobianBlockType>::getJs()
{
    if(!this->assemble.getValue() || !BlockType::constant || !eigenJacobian.rows()) updateJ();

    //    if( this->maskTo->isActivated() )
    //    {
    //        updateMaskedJ();
    //        baseMatrices[0] = &maskedEigenJacobian;
    //    }
    //    else
    //    {
    //        baseMatrices[0] = &eigenJacobian;
    //    }

    return &baseMatrices;
}

template <class JacobianBlockType>
const defaulttype::BaseMatrix* BaseDeformationMappingT<JacobianBlockType>::getK()
{
    if( BlockType::constant || !K.compressedMatrix.nonZeros() ) return NULL;
    else return &K;
}

template <class JacobianBlockType>
typename BaseDeformationMappingT<JacobianBlockType>::SparseMatrix& BaseDeformationMappingT<JacobianBlockType>::getJacobianBlocks()
{
    if(!this->assemble.getValue() || !BlockType::constant) updateJ();
    return jacobian;
}


template <class JacobianBlockType>
void BaseDeformationMappingT<JacobianBlockType>::updateForceMask()
{
    const VecVRef& indices = this->f_index.getValue();
    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->getEntry(i) )
        {
            for(size_t j=0; j<jacobian[i].size(); j++)
            {
                size_t index = indices[i][j];
                this->maskFrom->insertEntry( index );
            }
        }
    }

    //    serr<<"updateForceMask "<<this->maskTo->nbActiveDofs()<<" "<<this->maskFrom->nbActiveDofs()<<sendl;
}


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
