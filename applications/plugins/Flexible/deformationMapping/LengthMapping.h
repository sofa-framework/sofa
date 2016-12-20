/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_LengthMapping_H
#define SOFA_COMPONENT_MAPPING_LengthMapping_H

#include <Flexible/config.h>
#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vectorData.h>


namespace sofa
{

namespace component
{

namespace mapping
{

static const SReal s_null_distance_epsilon = 1e-8;




/** Maps point positions to length

@author Benjamin GILLES
  */

template <class TIn, class TOut>
class LengthMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LengthMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;
    typedef linearsolver::EigenSparseMatrix<TIn,TIn>     SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

    typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef typename sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::helper::vector< Index > VecIndex;

    Data<helper::vector<Real> > offset;
    Data<unsigned int> f_nbMeshes;
    helper::vectorData< SeqEdges > vf_edges;
    Data<bool> f_geometricStiffness; ///< should geometricStiffness be considered?

    virtual void init()
    {
        baseMatrices.resize( 1 );
        baseMatrices[0] = &jacobian;
        reinit();
        this->Inherit::init();
    }

    virtual void reinit()
    {
        vf_edges.resize(f_nbMeshes.getValue());
        this->getToModel()->resize( f_nbMeshes.getValue() );
        this->Inherit::reinit();
    }

    Real processEdge(const unsigned meshIndex, const unsigned a, const unsigned b, const InCoord A, const InCoord B, const bool useGeomStiffness)
    {
        InDeriv ab = B - A, dir;
        Real l = ab.norm();
        if(l<s_null_distance_epsilon) // -> arbitrary direction
        {
            Real p = 1.0f/std::sqrt((Real)Nin);
            for( unsigned i=0;i<Nin;++i) dir[i]=p;
        }
        else dir = ab/l;
        for(unsigned k=0; k<Nin; k++ )
        {
            jacobian.add(meshIndex, a*Nin+k, -dir[k] );
            jacobian.add(meshIndex, b*Nin+k, dir[k] );
        }

        if(l>s_null_distance_epsilon && useGeomStiffness )
        {
            Real dj;
            for(unsigned j=0; j<Nin; j++ )
                for(unsigned k=0; k<Nin; k++ )
                {
                    if(j==k) dj = 1 - dir[j]*dir[k];
                    else dj = - dir[j]*dir[k];
                    dj/=l;

                    hessian[meshIndex].add(a*Nin+j, a*Nin+k,  dj);
                    hessian[meshIndex].add(b*Nin+j, a*Nin+k, -dj);
                    hessian[meshIndex].add(a*Nin+j, b*Nin+k, -dj);
                    hessian[meshIndex].add(b*Nin+j, b*Nin+k,  dj);
                }
            hessian[meshIndex].compress();
        }
        return l;
    }

    virtual void apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        helper::WriteOnlyAccessor< Data<OutVecCoord> >  v = dOut;
        helper::ReadAccessor< Data<InVecCoord> >  x = dIn;
        helper::ReadAccessor< Data<helper::vector<Real> > > off = offset;
        bool useGeomStiffness = f_geometricStiffness.getValue();

        for(size_t m=0;m<f_nbMeshes.getValue();++m) v[m][0] = off[std::min(m,offset.getValue().size())];
        jacobian.resizeBlocks(v.size(),x.size());
        if( useGeomStiffness )
        {
            hessian.resize(v.size());
            for(size_t m=0;m<f_nbMeshes.getValue();++m) hessian[m].resizeBlocks(x.size(),x.size());
            K.resizeBlocks(x.size(),x.size());
        }

        for (size_t m = 0; m < f_nbMeshes.getValue(); ++m)
        {
            raEdges edges(*this->vf_edges[m]);
            for (size_t i = 0; i < edges.size(); ++i)
            {
                const Edge& t = edges[i];
                v[m][0] += processEdge(m,t[0],t[1],x[t[0]],x[t[1]],useGeomStiffness);
            }
        }

        jacobian.compress();
    }

    virtual void applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)    { if( jacobian.rowSize() > 0 ) jacobian.mult(dOut,dIn);    }
    virtual void applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)    { if( jacobian.rowSize() > 0 ) jacobian.addMultTranspose(dIn,dOut);    }
    virtual void applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& /*dIn*/, const Data<OutMatrixDeriv>& /*dOut*/) {}

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        if( !f_geometricStiffness.getValue() ) return;
        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);
        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);
        for(size_t m=0;m<f_nbMeshes.getValue();++m)        hessian[m].addMult(parentForceData,parentDisplacementData,mparams->kFactor()*childForce[m][0]);
    }

    virtual const defaulttype::BaseMatrix* getK()
    {
        if( f_geometricStiffness.getValue() )
        {
            const OutVecDeriv& childForce = this->toModel->readForces().ref();
            for(size_t m=0;m<f_nbMeshes.getValue();++m) K.compressedMatrix += hessian[m].compressedMatrix * childForce[m][0];
            K.compress();
        }
        return &K;
    }

    virtual const sofa::defaulttype::BaseMatrix* getJ() { return &jacobian; }
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs()    { return &baseMatrices; }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        vf_edges.parseSizeData(arg, f_nbMeshes);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        vf_edges.parseFieldsSizeData(str, f_nbMeshes);
        Inherit1::parseFields(str);
    }

protected:
    LengthMapping()
        : Inherit()
        , offset(initData(&offset, helper::vector<Real>((int)1,(Real)0.0), "offset", "offsets added to output lengths"))
        , f_nbMeshes( initData (&f_nbMeshes, (unsigned)1, "nbMeshes", "number of meshes to compute the length for") )
        , vf_edges(this,"edges", "input edges for mesh ")
        , f_geometricStiffness( initData( &f_geometricStiffness, false, "geometricStiffness", "Should geometricStiffness be considered?" ) )
    {
        vf_edges.resize(f_nbMeshes.getValue());
        this->addAlias(vf_edges[0], "edges");

    }

    virtual ~LengthMapping() {}

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    helper::vector<SparseKMatrixEigen> hessian;
    SparseKMatrixEigen K; ///< Stiffness due to the non-linearity of the mapping
};



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
