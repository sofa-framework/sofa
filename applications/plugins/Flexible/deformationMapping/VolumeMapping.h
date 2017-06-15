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
#ifndef SOFA_COMPONENT_MAPPING_VolumeMapping_H
#define SOFA_COMPONENT_MAPPING_VolumeMapping_H

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

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class VolumeMappingInternalData
{
public:
};


/** Maps point positions to volume

@author Benjamin GILLES
  */

template <class TIn, class TOut>
class VolumeMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(VolumeMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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

    typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef typename core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef helper::ReadAccessor<Data< SeqQuads > > raQuads;
    typedef typename sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::helper::vector< Index > VecIndex;

    virtual void init()
    {
        baseMatrices.resize( 1 );
        baseMatrices[0] = &jacobian;
        reinit();
        this->Inherit::init();
    }

    virtual void reinit()
    {
        vf_triangles.resize(f_nbMeshes.getValue());
        vf_quads.resize(f_nbMeshes.getValue());
        this->getToModel()->resize( f_nbMeshes.getValue() );
        this->Inherit::reinit();
    }

    Real processTriangle(const unsigned meshIndex, const unsigned a, const unsigned b, const unsigned c, const InCoord A, const InCoord B, const InCoord C)
    {
        InDeriv ab = B - A;
        InDeriv ac = C - A;
        InDeriv bc = C - B;
        InDeriv sn = (ab.cross(ac))/6.;

        for(unsigned k=0; k<Nin; k++ )
        {
            jacobian.add(meshIndex, a*Nin+k, sn[k] );
            jacobian.add(meshIndex, b*Nin+k, sn[k] );
            jacobian.add(meshIndex, c*Nin+k, sn[k] );
        }
        if( f_geometricStiffness.getValue() )
        {
            defaulttype::Mat<3,3,Real> DsnDA;
            DsnDA[0][0]=0;           DsnDA[0][1]=-bc[2]/6.;   DsnDA[0][2]=bc[1]/6.;
            DsnDA[1][0]=bc[2]/6.;    DsnDA[1][1]=0;           DsnDA[1][2]=-bc[0]/6.;
            DsnDA[2][0]=-bc[1]/6.;   DsnDA[2][1]=bc[0]/6.;    DsnDA[2][2]=0;

            defaulttype::Mat<3,3,Real> DsnDB;
            DsnDB[0][0]=0;           DsnDB[0][1]=ac[2]/6.;    DsnDB[0][2]=-ac[1]/6.;
            DsnDB[1][0]=-ac[2]/6.;   DsnDB[1][1]=0;           DsnDB[1][2]=ac[0]/6.;
            DsnDB[2][0]=ac[1]/6.;    DsnDB[2][1]=-ac[0]/6.;   DsnDB[2][2]=0;


            defaulttype::Mat<3,3,Real> DsnDC;
            DsnDC[0][0]=0;           DsnDC[0][1]=-ab[2]/6.;   DsnDC[0][2]=ab[1]/6.;
            DsnDC[1][0]=ab[2]/6.;    DsnDC[1][1]=0;           DsnDC[1][2]=-ab[0]/6.;
            DsnDC[2][0]=-ab[1]/6.;   DsnDC[2][1]=ab[0]/6.;    DsnDC[2][2]=0;

            for(unsigned j=0; j<Nin; j++ )
                for(unsigned k=0; k<Nin; k++ )
                {
                    hessian[meshIndex].add(a*Nin+j, a*Nin+k,  DsnDA[j][k] );
                    hessian[meshIndex].add(b*Nin+j, a*Nin+k,  DsnDA[j][k] );
                    hessian[meshIndex].add(c*Nin+j, a*Nin+k,  DsnDA[j][k] );

                    hessian[meshIndex].add(a*Nin+j, b*Nin+k,  DsnDB[j][k] );
                    hessian[meshIndex].add(b*Nin+j, b*Nin+k,  DsnDB[j][k] );
                    hessian[meshIndex].add(c*Nin+j, b*Nin+k,  DsnDB[j][k] );

                    hessian[meshIndex].add(a*Nin+j, c*Nin+k,  DsnDC[j][k] );
                    hessian[meshIndex].add(b*Nin+j, c*Nin+k,  DsnDC[j][k] );
                    hessian[meshIndex].add(c*Nin+j, c*Nin+k,  DsnDC[j][k] );
                }
        }
        return sn[2] * (A[2] + B[2] + C[2]);
    }

    virtual void apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        helper::WriteOnlyAccessor< Data<OutVecCoord> >  v = dOut;
        helper::ReadAccessor< Data<InVecCoord> >  x = dIn;

        for(size_t m=0;m<f_nbMeshes.getValue();++m) v[m][0] = m<offset.getValue().size()?offset.getValue()[m]:offset.getValue()[0];
        hessian.resize(v.size());
        for(size_t m=0;m<f_nbMeshes.getValue();++m) hessian[m].resizeBlocks(x.size(),x.size());
        K.resizeBlocks(x.size(),x.size());
        jacobian.resizeBlocks(v.size(),x.size());

        for (size_t m = 0; m < f_nbMeshes.getValue(); ++m)
        {
            raTriangles triangles(*this->vf_triangles[m]);
            raQuads quads(*this->vf_quads[m]);

            for (size_t i = 0; i < triangles.size(); ++i)
            {
                const Triangle& t = triangles[i];
                v[m][0] += processTriangle(m,t[0],t[1],t[2],x[t[0]],x[t[1]],x[t[2]]);
            }

            for (size_t i = 0; i < quads.size(); ++i)
            {
                const Quad& q = quads[i];
                v[m][0] += processTriangle(m,q[0],q[1],q[2],x[q[0]],x[q[1]],x[q[2]]);
                v[m][0] += processTriangle(m,q[0],q[2],q[3],x[q[0]],x[q[2]],x[q[3]]);
            }
            hessian[m].compress();
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
        vf_triangles.parseSizeData(arg, f_nbMeshes);
        vf_quads.parseSizeData(arg, f_nbMeshes);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        vf_triangles.parseFieldsSizeData(str, f_nbMeshes);
        vf_quads.parseFieldsSizeData(str, f_nbMeshes);
        Inherit1::parseFields(str);
    }

protected:
    VolumeMapping()
        : Inherit()
        , offset(initData(&offset, helper::vector<Real>((int)1,(Real)0.0), "offset", "offsets added to output volumes"))
        , f_nbMeshes( initData (&f_nbMeshes, (unsigned)1, "nbMeshes", "number of meshes to compute the volume for") )
        , vf_triangles(this,"triangles", "input triangles for mesh ")
        , vf_quads(this,"quads", "input quads for mesh ")
        , f_geometricStiffness( initData( &f_geometricStiffness, false, "geometricStiffness", "Should geometricStiffness be considered?" ) )
    {
        vf_triangles.resize(f_nbMeshes.getValue());
        vf_quads.resize(f_nbMeshes.getValue());
        this->addAlias(vf_triangles[0], "triangles");
        this->addAlias(vf_quads[0], "quads");

    }

    virtual ~VolumeMapping() {}

    Data<helper::vector<Real> > offset;

    Data<unsigned int> f_nbMeshes;
    helper::vectorData< SeqTriangles > vf_triangles;
    helper::vectorData< SeqQuads > vf_quads;
    Data<bool> f_geometricStiffness; ///< should geometricStiffness be considered?

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    helper::vector<SparseKMatrixEigen> hessian;
    SparseKMatrixEigen K; ///< Stiffness due to the non-linearity of the mapping
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_VolumeMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API VolumeMapping< Vec3dTypes, Vec1dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API VolumeMapping< Vec3fTypes, Vec1fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
