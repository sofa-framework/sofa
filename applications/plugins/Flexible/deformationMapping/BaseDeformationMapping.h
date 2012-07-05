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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H
#define SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H

#include "../initFlexible.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../types/AffineTypes.h"
#include "../types/QuadraticTypes.h"
#include "../types/DeformationGradientTypes.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/vector.h>

#include "../shapeFunction/BaseShapeFunction.h"
#include "../quadrature/BaseGaussPointSampler.h"
#include <sofa/component/topology/TopologyData.inl>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/vector.h>
#include <sofa/core/Mapping.inl>

#include <sofa/component/linearsolver/EigenSparseMatrix.h>

namespace sofa
{


template< class OutDataTypes>
class OutDataTypesInfo
{
public:
    enum {material_dimensions = OutDataTypes::material_dimensions};
    static const bool positionMapped=false; ///< tells if spatial positions are included in OutDataTypes (false for deformation gradients)
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
    static const bool positionMapped=true;
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::ExtVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
    static const bool positionMapped=true;
};

template<int TDim, class TReal>
class OutDataTypesInfo<defaulttype::StdAffineTypes<TDim, TReal> >
{
public:
    enum {material_dimensions = TDim};
    static const bool positionMapped=true;
};

template<int TDim, class TReal>
class OutDataTypesInfo<defaulttype::StdRigidTypes<TDim, TReal> >
{
public:
    enum {material_dimensions = TDim};
    static const bool positionMapped=true;
};

template<int TDim, class TReal>
class OutDataTypesInfo<defaulttype::StdQuadraticTypes<TDim, TReal> >
{
public:
    enum {material_dimensions = TDim};
    static const bool positionMapped=true;
};



///< drawing function for deformation gradients (handled by mapping since spatial positions are not included in state)
template< class OutDataTypes>
class DrawOut
{
public:
    typedef defaulttype::Vec<OutDataTypes::spatial_dimensions,typename OutDataTypes::Real> sCoord ;
    static void draw(const core::visual::VisualParams* , const sCoord& , const typename OutDataTypes::Coord&, const float&, const defaulttype::Vec<4,float>&) {}
};


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class DrawOut<defaulttype::DefGradientTypes<_spatial_dimensions, _material_dimensions, _order, _Real> >
{
public:
    typedef defaulttype::DefGradientTypes<_spatial_dimensions, _material_dimensions, _order, _Real> OutDataTypes;
    typedef defaulttype::Vec<OutDataTypes::spatial_dimensions,typename OutDataTypes::Real> sCoord ;
    static void draw(const core::visual::VisualParams* vparams, const sCoord& p, const typename OutDataTypes::Coord& x, const float& scale, const defaulttype::Vec<4,float>& colour)
    {
        for(int i=0; i<_material_dimensions; i++)
        {
            sCoord u=x.getF().transposed()(i)*0.5*scale;
            vparams->drawTool()->drawCylinder(p-u,p+u,0.02*scale,colour,4);
        }
    }
};


namespace component
{
namespace mapping
{

using helper::vector;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class BaseDeformationMappingInternalData
{
public:
};


/** Abstract mapping (one parent->several children with different influence) using JacobianBlocks or sparse eigen matrix
*/

template <class JacobianBlockType>
class SOFA_Flexible_API BaseDeformationMapping : public core::Mapping<typename JacobianBlockType::In,typename JacobianBlockType::Out>
{
public:
    typedef core::Mapping<typename JacobianBlockType::In, typename JacobianBlockType::Out> Inherit;
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(BaseDeformationMapping,JacobianBlockType), SOFA_TEMPLATE2(core::Mapping,typename JacobianBlockType::In,typename JacobianBlockType::Out));

    /** @name  Input types    */
    //@{
    typedef typename JacobianBlockType::In In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real Real;
    //@}

    /** @name  Output types    */
    //@{
    typedef typename JacobianBlockType::Out Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    enum { spatial_dimensions = Out::spatial_dimensions };
    enum { material_dimensions = OutDataTypesInfo<Out>::material_dimensions };
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef core::behavior::ShapeFunctionTypes<material_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::VGradient VGradient;
    typedef typename BaseShapeFunction::VHessian VHessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::MaterialToSpatial MaterialToSpatial ; ///< deformation gradient type
    typedef typename BaseShapeFunction::VMaterialToSpatial VMaterialToSpatial;
    typedef typename BaseShapeFunction::Coord mCoord; ///< material coordinates
    //@}

    /** @name  Coord types    */
    //@{
    typedef Vec<spatial_dimensions,Real> Coord ; ///< spatial coordinates
    typedef vector<Coord> VecCoord;
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef JacobianBlockType BlockType;
    typedef vector<vector<BlockType> >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;

    typedef typename BlockType::KBlock  KBlock;  ///< stiffness block matrix
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    //@}


    virtual void resizeOut()
    {
        if(this->f_printLog.getValue()) std::cout<<"deformationMapping::resizeOut()"<<std::endl;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

        helper::WriteAccessor<Data<VecCoord> > pos0 (this->f_pos0);
        this->mapPosition0Needed=true; // need to update mapped spatial positions if needed for visualization

        //VecCoord pos0;

        unsigned int size;

        if( !_sampler ) this->getContext()->get(_sampler,core::objectmodel::BaseContext::Local);
        if(_sampler) // retrieve initial positions from gauss point sampler (deformation gradient types)
        {
            size =_sampler->getNbSamples();
            this->toModel->resize(size);
            pos0.resize(size);  for(unsigned int i=0; i<size; i++) pos0[i]=_sampler->getSample(i);
            if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< size <<" gauss points imported"<<std::endl;
        }
        else  // retrieve initial positions from children dofs (vec types)
        {
            size = out.size();
            pos0.resize(size);  for(unsigned int i=0; i<size; i++ )  Out::get(pos0[i][0],pos0[i][1],pos0[i][2],out[i]);
        }

        // init shape function
        if( !_shapeFunction ) this->getContext()->get(_shapeFunction,core::objectmodel::BaseContext::SearchUp);
        if ( !_shapeFunction ) serr << "ShapeFunction<"<<ShapeFunctionType::Name()<<"> component not found" << sendl;
        else
        {
            vector<mCoord> mpos0;
            mpos0.resize(pos0.size());
            for(unsigned int i=0; i<pos0.size(); ++i)  StdVectorTypes<mCoord,mCoord>::set( mpos0[i], pos0[i][0] , pos0[i][1] , pos0[i][2]);
            if(_sampler)   // get weights associated to gauss point regions
                _shapeFunction->computeShapeFunction(mpos0,*this->f_M.beginEdit(),*this->f_index.beginEdit(),*this->f_w.beginEdit(),*this->f_dw.beginEdit(),*this->f_ddw.beginEdit(),_sampler->getRegion());
            else            // interpolate weights at sample positions
                _shapeFunction->computeShapeFunction(mpos0,*this->f_M.beginEdit(),*this->f_index.beginEdit(),*this->f_w.beginEdit(),*this->f_dw.beginEdit(),*this->f_ddw.beginEdit());
            this->f_index.endEdit();     this->f_M.endEdit();    this->f_w.endEdit();        this->f_dw.endEdit();        this->f_ddw.endEdit();
        }

        // init jacobians
        jacobian.resize(size);
        for(unsigned int i=0; i<size; i++ )
        {
            unsigned int nbref=this->f_index.getValue()[i].size();
            jacobian[i].resize(nbref);
            for(unsigned int j=0; j<nbref; j++ )
            {
                unsigned int index=this->f_index.getValue()[i][j];
                jacobian[i][j].init( in[index],out[i],pos0[i],f_M.getValue()[i],f_w.getValue()[i][j],f_dw.getValue()[i][j],f_ddw.getValue()[i][j]);
            }
        }

        reinit();
    }


    /** @name Mapping functions */
    //@{

    virtual void init()
    {
        if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
            maskFrom = &stateFrom->forceMask;
        if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
            maskTo = &stateTo->forceMask;


        baseMatrices.resize( 1 ); // just a wrapping for getJs()
        baseMatrices[0] = &eigenJacobian;

        resizeOut();

        Inherit::init();
    }

    virtual void reinit()
    {
        if(this->assembleJ.getValue()) updateJ();

        Inherit::reinit();
    }

    virtual void apply(const core::MechanicalParams */*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        helper::ReadAccessor<Data<OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
        if(_sampler) if(_sampler->getNbSamples()!=outpos.size()) resizeOut();

        OutVecCoord&  out = *dOut.beginEdit();
        const InVecCoord&  in = dIn.getValue();

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            out[i]=OutCoord();
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                unsigned int index=this->f_index.getValue()[i][j];
                jacobian[i][j].addapply(out[i],in[index]);
            }
        }
        dOut.endEdit();

        if(!BlockType::constantJ) if(this->assembleJ.getValue()) updateJ();

        this->mapPosition0Needed=true; // need to update mapped spatial positions if needed for visualization
    }



    virtual void applyJ(const core::MechanicalParams */*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
    {
        if(this->assembleJ.getValue())  eigenJacobian.mult(dOut,dIn);
        else
        {
            OutVecDeriv&  out = *dOut.beginEdit();
            const InVecDeriv&  in = dIn.getValue();

            if ((!this->maskTo)||(this->maskTo&& !(this->maskTo->isInUse())) )
            {
                for(unsigned int i=0; i<jacobian.size(); i++)
                {
                    out[i]=OutDeriv();
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addmult(out[i],in[index]);
                    }
                }
            }
            else
            {
                typedef helper::ParticleMask ParticleMask;
                const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
                for (ParticleMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
                {
                    unsigned int i= ( unsigned int ) ( *it );
                    out[i]=OutDeriv();
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addmult(out[i],in[index]);
                    }
                }
            }

            dOut.endEdit();
        }
    }


    virtual void applyJT(const core::MechanicalParams */*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
    {
        if(this->assembleJ.getValue())  eigenJacobian.addMultTranspose(dIn,dOut);
        else
        {
            InVecDeriv&  in = *dIn.beginEdit();
            const OutVecDeriv&  out = dOut.getValue();

            if((!this->maskTo)||(this->maskTo&& !(this->maskTo->isInUse())) )
            {
                for(unsigned int i=0; i<jacobian.size(); i++)
                {
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addMultTranspose(in[index],out[i]);
                    }
                }
            }
            else
            {
                typedef helper::ParticleMask ParticleMask;
                const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
                for (ParticleMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
                {
                    const int i= ( int ) ( *it );
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addMultTranspose(in[index],out[i]);
                    }
                }
            }

            dIn.endEdit();
        }
    }

    virtual void applyJT(const core::ConstraintParams */*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/)
    {

    }

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        if(BlockType::constantJ) return;

        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);

        helper::WriteAccessor<Data<InVecDeriv> > parentForce (parentForceData);
        helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (parentDisplacementData);
        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);

        if(this->assembleK.getValue())
        {
            updateK(childForce.ref());
            K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
        }
        else
        {
            if((!this->maskTo)||(this->maskTo&& !(this->maskTo->isInUse())) )
            {
                for(unsigned int i=0; i<jacobian.size(); i++)
                {
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addDForce( parentForce[index], parentDisplacement[index], childForce[i], mparams->kFactor() );
                    }
                }
            }
            else
            {
                typedef helper::ParticleMask ParticleMask;
                const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
                for (ParticleMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
                {
                    const int i= ( int ) ( *it );
                    for(unsigned int j=0; j<jacobian[i].size(); j++)
                    {
                        unsigned int index=this->f_index.getValue()[i][j];
                        jacobian[i][j].addDForce( parentForce[index], parentDisplacement[index], childForce[i], mparams->kFactor() );
                    }
                }
            }
        }
    }

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams */*mparams*/)
    {
        if(!this->assembleJ.getValue()) updateJ();
        return &eigenJacobian;
    }

    // Compliant plugin experimental API
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs() { return &baseMatrices; }


    void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowMechanicalMappings() && !showObjectScale.getValue()) return;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<vector<VRef> > > ref (this->f_index);
        helper::ReadAccessor<Data<vector<VReal> > > w (this->f_w);

        if(this->mapPosition0Needed && !OutDataTypesInfo<Out>::positionMapped) {mapPosition0(); this->mapPosition0Needed=false; }

        if (vparams->displayFlags().getShowMechanicalMappings())
        {
            vector< defaulttype::Vec3d > edge;     edge.resize(2);
            Vec<4,float> col;

            for(unsigned i=0; i<ref.size(); i++ )
            {
                if(OutDataTypesInfo<Out>::positionMapped) Out::get(edge[1][0],edge[1][1],edge[1][2],out[i]);
                else edge[1]=f_pos[i];
                for(unsigned j=0; j<ref[i].size(); j++ )
                    if(w[i][j])
                    {
                        In::get(edge[0][0],edge[0][1],edge[0][2],in[ref[i][j]]);
                        sofa::helper::gl::Color::getHSVA(&col[0],240.*w[i][j],1.,.8,1.);
                        vparams->drawTool()->drawLines ( edge, 1, col );
                    }
            }
        }
        if (showObjectScale.getValue())
        {
            if(!OutDataTypesInfo<Out>::positionMapped) // when out does not contain positions (e.g. deformation gradients), drawing is handled by the mapper (instead of mechanical object)
            {
//                glPushAttrib ( GL_LIGHTING_BIT );
//                glDisable ( GL_LIGHTING );
                float scale=showObjectScale.getValue();
                Vec<4,float> col( 1.0, 1.0, 0.0, 1.0 );
                for(unsigned i=0; i<ref.size(); i++ ) DrawOut<Out>::draw(vparams,f_pos[i],out[i],scale, col);
//                glPopAttrib();
            }
        }
    }

    //@}



    SparseMatrix& getJacobianBlocks() { return jacobian; }


    engine::BaseGaussPointSampler* _sampler;
    BaseShapeFunction* _shapeFunction;        ///< where the weights are computed
    Data<vector<VRef> > f_index;            ///< The numChildren * numRefs column indices. index[i][j] is the index of the j-th parent influencing child i.
    Data<vector<VReal> >       f_w;
    Data<vector<VGradient> >   f_dw;
    Data<vector<VHessian> >    f_ddw;
    Data<VMaterialToSpatial>    f_M;

protected:
    BaseDeformationMapping (core::State<In>* from = NULL, core::State<Out>* to= NULL)
        : Inherit ( from, to )
        , _sampler(NULL)
        , _shapeFunction(NULL)
        , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
        , f_w ( initData ( &f_w,"weights","influence weights of the Dofs" ) )
        , f_dw ( initData ( &f_dw,"weightGradients","weight gradients" ) )
        , f_ddw ( initData ( &f_ddw,"weightHessians","weight Hessians" ) )
        , f_M ( initData ( &f_M,"M","Transformations from material to 3d space (linear for now..)" ) )
        , f_pos0 ( initData ( &f_pos0,"restPosition","initial spatial positions of children" ) )
        , mapPosition0Needed(true)
        , maskFrom(NULL)
        , maskTo(NULL)
        , assembleJ ( initData ( &assembleJ,false, "assembleJ","Assemble the Jacobian matrix or use optimized Jacobian/vector multiplications" ) )
        , assembleK ( initData ( &assembleK,false, "assembleK","Assemble the geometric stiffness matrix or use optimized Jacobian/vector multiplications" ) )
        , showObjectScale(initData(&showObjectScale, (float)0.0, "showObjectScale", "Scale for deformation gradient display"))
    {
        if(OutDataTypesInfo<Out>::positionMapped)
        {
            showObjectScale.setDisplayed(false);
        }
    }

    virtual ~BaseDeformationMapping()     { }

    Data<VecCoord >    f_pos0; ///< initial spatial positions of children
    virtual void mapPosition0()=0;   ///< map initial spatial positions stored in f_pos0 to f_pos (used to display deformation gradients)
    VecCoord f_pos;
    bool mapPosition0Needed;  // used to speed up drawing

    SparseMatrix jacobian;   ///< Jacobian of the mapping

    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements

    Data<bool> assembleJ;
    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API
    void updateJ()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        //helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        eigenJacobian.resizeBlocks(jacobian.size(),in.size());
        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            //            vector<MatBlock> blocks;
            //            vector<unsigned> columns;
            eigenJacobian.beginBlockRow(i);
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                //                columns.push_back( this->f_index.getValue()[i][j] );
                //                blocks.push_back( jacobian[i][j].getJ() );
                eigenJacobian.createBlock( this->f_index.getValue()[i][j], jacobian[i][j].getJ());
            }
            //            eigenJacobian.appendBlockRow( i, columns, blocks );
            eigenJacobian.endBlockRow();
        }
        //        eigenJacobian.endEdit();
        eigenJacobian.compress();
    }

    Data<bool> assembleK;
    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    void updateK(const OutVecDeriv& childForce)
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        K.resizeBlocks(in.size(),in.size());
        vector<KBlock> diagonalBlocks; diagonalBlocks.resize(in.size());

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                unsigned int index=this->f_index.getValue()[i][j];
                diagonalBlocks[index] += jacobian[i][j].getK(childForce[i]);
            }
        }
        for(unsigned int i=0; i<in.size(); i++)
        {
            //            vector<KBlock> blocks;
            //            vector<unsigned> columns;
            //            columns.push_back( i );
            //            blocks.push_back( diagonalBlocks[i] );
            //            K.appendBlockRow( i, columns, blocks );
            K.beginBlockRow(i);
            K.createBlock(i,diagonalBlocks[i]);
            K.endBlockRow();
        }
        //        K.endEdit();
        K.compress();
    }

    Data< float > showObjectScale;

};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
