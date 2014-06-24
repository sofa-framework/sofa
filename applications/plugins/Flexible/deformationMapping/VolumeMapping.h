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
*                               SOFA :: Modules                               *
*                                                                             *
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


namespace sofa
{
using helper::vector;

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

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::helper::vector< Index > VecIndex;

    virtual void init()
    {
        m_topology = this->getContext()->getMeshTopology();
        if( !m_topology ) serr<<"No MeshTopology found ! "<<sendl;

        this->getToModel()->resize( 1 );
        baseMatrices.resize( 1 );
        baseMatrices[0] = &jacobian;

        this->Inherit::init();
    }

    Real processTriangle(const unsigned a, const unsigned b, const unsigned c, const InCoord A, const InCoord B, const InCoord C)
    {
        InDeriv ab = B - A;
        InDeriv ac = C - A;
        InDeriv bc = C - B;
        InDeriv sn = (ab.cross(ac))/6.;

        for(unsigned k=0; k<Nin; k++ )
        {
            jacobian.add(0, a*Nin+k, sn[k] );
            jacobian.add(0, b*Nin+k, sn[k] );
            jacobian.add(0, c*Nin+k, sn[k] );
        }

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
                geometricStiffness.add(a*Nin+j, a*Nin+k,  DsnDA[j][k] );
                geometricStiffness.add(b*Nin+j, a*Nin+k,  DsnDA[j][k] );
                geometricStiffness.add(c*Nin+j, a*Nin+k,  DsnDA[j][k] );

                geometricStiffness.add(a*Nin+j, b*Nin+k,  DsnDB[j][k] );
                geometricStiffness.add(b*Nin+j, b*Nin+k,  DsnDB[j][k] );
                geometricStiffness.add(c*Nin+j, b*Nin+k,  DsnDB[j][k] );

                geometricStiffness.add(a*Nin+j, c*Nin+k,  DsnDC[j][k] );
                geometricStiffness.add(b*Nin+j, c*Nin+k,  DsnDC[j][k] );
                geometricStiffness.add(c*Nin+j, c*Nin+k,  DsnDC[j][k] );
            }

        return sn[2] * (A[2] + B[2] + C[2]);
    }

    virtual void apply(const core::MechanicalParams */*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        if(!m_topology) return;

        helper::WriteAccessor< Data<OutVecCoord> >  v = dOut;
        helper::ReadAccessor< Data<InVecCoord> >  x = dIn;

        v[0][0] = offset.getValue();
        jacobian.resizeBlocks(v.size(),x.size());
        geometricStiffness.resizeBlocks(x.size(),x.size());

        for (int i = 0; i < m_topology->getNbTriangles(); i++)
        {
            Triangle t = m_topology->getTriangle(i);
            v[0][0] += processTriangle(t[0],t[1],t[2],x[t[0]],x[t[1]],x[t[2]]);
        }

        for (int i = 0; i < m_topology->getNbQuads(); i++)
        {
            Quad q = m_topology->getQuad(i);
            v[0][0] += processTriangle(q[0],q[1],q[2],x[q[0]],x[q[1]],x[q[2]]);
            v[0][0] += processTriangle(q[0],q[2],q[3],x[q[0]],x[q[2]],x[q[3]]);
        }

        jacobian.compress();
        geometricStiffness.compress();
    }

    virtual void applyJ(const core::MechanicalParams */*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)    { if( jacobian.rowSize() > 0 ) jacobian.mult(dOut,dIn);    }
    virtual void applyJT(const core::MechanicalParams */*mparams*/, Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)    { if( jacobian.rowSize() > 0 ) jacobian.addMultTranspose(dIn,dOut);    }
    virtual void applyJT(const core::ConstraintParams */*cparams*/, Data<InMatrixDeriv>& /*dIn*/, const Data<OutMatrixDeriv>& /*dOut*/) {}

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);
        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);
        geometricStiffness.addMult(parentForceData,parentDisplacementData,mparams->kFactor()*childForce[0][0]);
    }


    virtual const sofa::defaulttype::BaseMatrix* getJ() { return &jacobian; }
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs()    { return &baseMatrices; }


protected:
    VolumeMapping()
        : Inherit(),
          m_topology(NULL),
         offset(initData(&offset, (Real)0.0, "offset", "offset added to output volume"))
    {
    }

    virtual ~VolumeMapping() {}

    sofa::core::topology::BaseMeshTopology* m_topology;  ///< where the triangles/quads are defined
    Data<Real> offset;

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    SparseKMatrixEigen geometricStiffness;               ///< Stiffness due to the non-linearity of the mapping
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
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
