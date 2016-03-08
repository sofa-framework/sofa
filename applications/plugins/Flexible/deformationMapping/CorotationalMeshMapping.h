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
#ifndef SOFA_COMPONENT_MAPPING_CorotationalMeshMapping_H
#define SOFA_COMPONENT_MAPPING_CorotationalMeshMapping_H

#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/decompose.h>

#include <sofa/helper/GenerateRigid.h>


#ifdef _OPENMP
#include <omp.h>
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class CorotationalMeshMappingInternalData
{
public:
};


/** Rigidly aligns positions to rest positions for each element
  An output (uncompatible) mesh is generated based on duplicated nodes
  Corotational elasticity is obtained by applying linear forcefield to the output.

@author Benjamin GILLES
  */

template <class TIn, class TOut>
class CorotationalMeshMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CorotationalMeshMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename In::VecCoord VecCoord;
    typedef typename In::VecDeriv VecDeriv;
    typedef typename In::Coord Coord;
    typedef typename In::Deriv Deriv;
    typedef typename In::MatrixDeriv MatrixDeriv;
    typedef typename In::Real Real;
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;
    typedef linearsolver::EigenSparseMatrix<TIn,TIn>     SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };


    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;
//    typedef core::topology::BaseMeshTopology::Triangle Triangle;
//    typedef core::topology::BaseMeshTopology::Quad Quad;
//    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;
//    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
//    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;
//    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;

    typedef core::topology::BaseMeshTopology::PointID ID;
    typedef helper::vector<ID> VecID;
    typedef helper::vector<VecID> VecVecID;

    typedef core::topology::BaseMeshTopology::index_type index_type;
    typedef helper::vector< index_type > VecIndex;

    typedef defaulttype::Mat<3,3,Real> Mat3x3;

    virtual void init()
    {
        this->getToModel()->resize( 1 );
        baseMatrices.resize( 1 );
        baseMatrices[0] = &jacobian;

        helper::ReadAccessor<Data<VecCoord> > pos0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));
        helper::ReadAccessor<Data< SeqTetrahedra > > rtetrahedra(this->in_tetrahedra);
        helper::ReadAccessor<Data< SeqHexahedra > > rhexahedra(this->in_hexahedra);
//        helper::ReadAccessor<Data< SeqTriangles > > rtriangles(this->in_triangles);
//        helper::ReadAccessor<Data< SeqQuads > > rquads(this->in_quads);
//        helper::ReadAccessor<Data< SeqEdges > > redges(this->in_edges);

        helper::WriteOnlyAccessor<Data< SeqTetrahedra > > wtetrahedra(this->out_tetrahedra); wtetrahedra.resize(0);
        helper::WriteOnlyAccessor<Data< SeqHexahedra > > whexahedra(this->out_hexahedra); whexahedra.resize(0);
//        helper::WriteOnlyAccessor<Data< SeqTriangles > > wtriangles(this->out_triangles); wtriangles.resize(0);
//        helper::WriteOnlyAccessor<Data< SeqQuads > > wquads(this->out_quads); wquads.resize(0);
//        helper::WriteOnlyAccessor<Data< SeqEdges > > wedges(this->out_edges); wedges.resize(0);

        this->index_parentToChild.resize(pos0.size());
        size_t nbOut=0;
        for (unsigned int i=0; i<rtetrahedra.size(); i++ )   { this->clusters.push_back(VecID());  this->clusters_child.push_back(VecID()); Tetrahedron cell; for (unsigned int j=0; j<4; j++ ) {this->clusters.back().push_back(rtetrahedra[i][j]); this->clusters_child.back().push_back(nbOut); this->index_childToParent.push_back(rtetrahedra[i][j]);  this->index_parentToChild[rtetrahedra[i][j]].push_back(nbOut);  cell[j]=nbOut; nbOut++;} wtetrahedra.push_back(cell); }
        for (unsigned int i=0; i<rhexahedra.size(); i++ )    { this->clusters.push_back(VecID());  this->clusters_child.push_back(VecID()); Hexahedron cell;  for (unsigned int j=0; j<8; j++ ) {this->clusters.back().push_back(rhexahedra[i][j]);  this->clusters_child.back().push_back(nbOut); this->index_childToParent.push_back(rhexahedra[i][j]);   this->index_parentToChild[rhexahedra[i][j]].push_back(nbOut);   cell[j]=nbOut; nbOut++;} whexahedra.push_back(cell); }
//        for (unsigned int i=0; i<rtriangles.size(); i++ )    { this->clusters.push_back(VecID());  this->clusters_child.push_back(VecID()); Triangle cell;    for (unsigned int j=0; j<3; j++ ) {this->clusters.back().push_back(rtriangles[i][j]);  this->clusters_child.back().push_back(nbOut); this->index_childToParent.push_back(rtriangles[i][j]);   this->index_parentToChild[rtriangles[i][j]].push_back(nbOut);   cell[j]=nbOut; nbOut++;} wtriangles.push_back(cell); }
//        for (unsigned int i=0; i<rquads.size(); i++ )        { this->clusters.push_back(VecID());  this->clusters_child.push_back(VecID()); Quad cell;        for (unsigned int j=0; j<4; j++ ) {this->clusters.back().push_back(rquads[i][j]);      this->clusters_child.back().push_back(nbOut); this->index_childToParent.push_back(rquads[i][j]);       this->index_parentToChild[rquads[i][j]].push_back(nbOut);       cell[j]=nbOut; nbOut++;} wquads.push_back(cell); }
//        for (unsigned int i=0; i<redges.size(); i++ )        { this->clusters.push_back(VecID());  this->clusters_child.push_back(VecID()); Edge cell;        for (unsigned int j=0; j<2; j++ ) {this->clusters.back().push_back(redges[i][j]);      this->clusters_child.back().push_back(nbOut); this->index_childToParent.push_back(redges[i][j]);       this->index_parentToChild[redges[i][j]].push_back(nbOut);       cell[j]=nbOut; nbOut++;} wedges.push_back(cell); }
        this->toModel->resize(nbOut);

        helper::WriteOnlyAccessor<Data<VecCoord> > pos (*this->toModel->write(core::VecCoordId::restPosition()));
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {
            for (unsigned int j=0; j<this->clusters_child[i].size() ; ++j)
            {
                ID pindex=this->clusters_child[i][j];
                pos[pindex] = pos0[this->index_childToParent[pindex]];
            }
        }



//        if( i==860 )
//        {
//            Coord mean = this->Xcm0[860];
//            Mat3x3 covariance;
//            for (size_t k = 0; k < this->clusters[860].size(); k++)
//            {
//                const Coord& point = pos0[this->clusters[860][k]];
//                for (int l = 0; l < 3; l++)
//                {
//                    for (int j = 0; j < 3; j++)
//                    {
//                        covariance[l][j] += (point[l]-mean[l]) * (point[j]-mean[j]);
//                    }
//                }
//            }
//            covariance /= this->clusters[860].size() - 1;
//            Mat3x3 rot;
//            helper::Decompose<Real>::polarDecomposition(covariance, rot);
//            serr<<"init "<<covariance<<"    "<<rot<<sendl;
//        }


//        int i = 860;
        //        Mat3x3 M;
//        for (unsigned int j=0; j<this->clusters[i].size() ; ++j)
//        {
//            ID pindex=this->clusters[i][j];
//            M += defaulttype::dyad(pos0[pindex],pos0[pindex]);
//        }

//        M -= defaulttype::dyad(this->Xcm0[i],this->Xcm0[i]); // sum wi.(X0-Xcm0)(X-Xcm)^T = sum wi.X0.X^T - Xcm0.sum(wi.X)^T
//        Mat3x3 rot;
//        helper::Decompose<Real>::polarDecomposition_stable(M, rot);
//        serr<<"init "<<M<<"    "<<rot<<sendl;


//        defaulttype::Rigid3Mass mass;
//        defaulttype::Vector3 center;
//        sofa::helper::io::Mesh mesh;
//        size_t size = this->clusters[i].size();
//        mesh.getVertices().resize(size);
//        for (unsigned int j=0; j<this->clusters[i].size() ; ++j)
//            mesh.getVertices()[j] = pos0[this->clusters[i][j]];

//        mesh.getFacets().resize(4);
//        mesh.getFacets()[0].resize(1); mesh.getFacets()[0][0].resize(3); mesh.getFacets()[0][0][0] = 0;mesh.getFacets()[0][0][1] = 1;mesh.getFacets()[0][0][2] = 2;
//        mesh.getFacets()[1].resize(1); mesh.getFacets()[1][0].resize(3); mesh.getFacets()[1][0][0] = 0;mesh.getFacets()[1][0][1] = 3;mesh.getFacets()[1][0][2] = 1;
//        mesh.getFacets()[2].resize(1); mesh.getFacets()[2][0].resize(3); mesh.getFacets()[2][0][0] = 1;mesh.getFacets()[2][0][1] = 3;mesh.getFacets()[2][0][2] = 2;
//        mesh.getFacets()[3].resize(1); mesh.getFacets()[3][0].resize(3); mesh.getFacets()[3][0][0] = 0;mesh.getFacets()[3][0][1] = 2;mesh.getFacets()[3][0][2] = 3;

//        helper::generateRigid(mass, center,  &mesh);
//        serr<<mass.inertiaMatrix/mass.mass<<sendl;

//        static const Coord restetra[4] = { Coord(-1,0,-1.0/std::sqrt(2.0)),Coord(1,0,-1.0/std::sqrt(2.0)),Coord(0,1,1.0/std::sqrt(2.0)),Coord(0,-1,1.0/std::sqrt(2.0)) };

//        Mat3x3 M, rot;
//        for (unsigned int j=0; j<this->clusters[i].size() ; ++j)
//        {
//            ID pindex=this->clusters[i][j];
//            M += defaulttype::dyad(restetra[j],pos0[pindex]);

//            serr<< (restetra[j]-restetra[(j+1)%4]).norm() <<sendl;
//        }

//        M -= defaulttype::dyad(Coord(0,0,0),this->Xcm0[i]); // sum wi.(X0-Xcm0)(X-Xcm)^T = sum wi.X0.X^T - Xcm0.sum(wi.X)^T
//        helper::Decompose<Real>::polarDecomposition(M, rot);
//        serr<<"init "<<rot<<sendl;


        rot0.resize(this->clusters.size());

        this->Xcm0.resize(this->clusters.size());
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {

            if( this->clusters[i].size() == 4 ) // tetra
            {
                Mat3x3 A;
                A[0] = pos0[this->clusters[i][1]]-pos0[this->clusters[i][0]];
                A[1] = pos0[this->clusters[i][2]]-pos0[this->clusters[i][0]];
                A[2] = pos0[this->clusters[i][3]]-pos0[this->clusters[i][0]];
                helper::Decompose<Real>::polarDecomposition( A, rot0[i] );

//                for(int j=0;j<4;++j)
//                {
//                    ID pindex=this->clusters_child[i][j];
//                    pos[pindex] = rot0[i] * pos0[this->index_childToParent[pindex]];
//                }
            }
        }

//        for (unsigned int i=0; i<rtetrahedra.size(); i++ )
//        {
//            const VecCoord &initialPoints=_initialPoints.getValue();
//            Mat3x3 A;
//            A[0] = initialPoints[b]-initialPoints[a];
//            A[1] = initialPoints[c]-initialPoints[a];
//            A[2] = initialPoints[d]-initialPoints[a];
//            //_initialTransformation[i] = A;

//            Transformation R_0_1;
//            helper::Decompose<Real>::polarDecomposition( A, R_0_1 );
//        }
//        for (unsigned int i=0; i<rhexahedra.size(); i++ )

        serr<<"init "<<rot0[860]<<sendl;




        this->Xcm0.resize(this->clusters.size());
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {
            for (unsigned int j=0; j<this->clusters[i].size() ; ++j) this->Xcm0[i]+=pos0[this->clusters[i][j]];
            this->Xcm0[i]/=(Real)this->clusters[i].size();
        }

        this->Inherit::init();
    }

    virtual void apply(const core::MechanicalParams */*mparams*/, Data<VecCoord>& dOut, const Data<VecCoord>& dIn)
    {
        helper::WriteOnlyAccessor< Data<VecCoord> >  posOut = dOut;
        helper::ReadAccessor< Data<VecCoord> >  pos = dIn;
        helper::ReadAccessor<Data<VecCoord> > pos0 (*this->fromModel->read(core::ConstVecCoordId::restPosition()));

        rot.resize(this->clusters.size());

        //#ifdef _OPENMP
        //        #pragma omp parallel for
        //#endif
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {
            Mat3x3 M;
            Coord Xcm;
            for (unsigned int j=0; j<this->clusters[i].size() ; ++j)
            {
                ID pindex=this->clusters[i][j];
                Xcm+=pos[pindex];
                M += defaulttype::dyad(pos0[pindex],pos[pindex]);
            }

            M -= defaulttype::dyad(this->Xcm0[i],Xcm); // sum wi.(X0-Xcm0)(X-Xcm)^T = sum wi.X0.X^T - Xcm0.sum(wi.X)^T
            helper::Decompose<Real>::polarDecomposition(M, this->rot[i]);

//            rot[i] = rot0[i] * rot[i] ;

            Coord tr = this->Xcm0[i] - this->rot[i] * Xcm/(Real)this->clusters[i].size();
            for (unsigned int j=0; j<this->clusters_child[i].size() ; ++j)
            {
                ID pindex=this->clusters_child[i][j];
                posOut[pindex] = this->rot[i] * pos[this->index_childToParent[pindex]]  + tr;
            }

//            if( i==860 )
//            {
//                helper::fixed_array<Coord,4> D;
//                for (unsigned int j=0; j<this->clusters_child[i].size() ; ++j)
//                {
//                    ID pindex=this->clusters_child[i][j];
//                    D[j]= pos[this->index_childToParent[pindex]] - posOut[pindex] ;
//                }
//                serr<<"rot "<<this->rot[i]<<sendl;
//                serr<<"D "<<D<<sendl;
//            }
        }
    }

    virtual void applyJ(const core::MechanicalParams */*mparams*/, Data<VecDeriv>& dOut, const Data<VecDeriv>& dIn)
    {

        helper::WriteOnlyAccessor< Data<VecDeriv> >  vOut = dOut;
        helper::ReadAccessor< Data<VecDeriv> >  v = dIn;
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {
            for (unsigned int j=0; j<this->clusters_child[i].size() ; ++j)
            {
                ID pindex=this->clusters_child[i][j];
                vOut[pindex] = this->rot[i] * v[this->index_childToParent[pindex]];
            }
        }
    }
    virtual void applyJT(const core::MechanicalParams */*mparams*/, Data<VecDeriv>& dIn, const Data<VecDeriv>& dOut)
    {
        helper::ReadAccessor< Data<VecDeriv> >  vOut = dOut;
        helper::WriteAccessor< Data<VecDeriv> >  v = dIn;
        for (unsigned int i=0 ; i<this->clusters.size() ; ++i)
        {
            for (unsigned int j=0; j<this->clusters_child[i].size() ; ++j)
            {
                ID pindex=this->clusters_child[i][j];
                v[this->index_childToParent[pindex]] += this->rot[i].multTranspose(vOut[pindex]);
            }
        }
    }
    virtual void applyJT(const core::ConstraintParams */*cparams*/, Data<MatrixDeriv>& /*dIn*/, const Data<MatrixDeriv>& /*dOut*/) {}

    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*parentDfId*/, core::ConstMultiVecDerivId )
    {
        //        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        //        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        //        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);
        //        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);
        //        geometricStiffness.addMult(parentForceData,parentDisplacementData,mparams->kFactor()*childForce[0][0]);
    }


    virtual const sofa::defaulttype::BaseMatrix* getJ() { return &jacobian; }
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs()    { return &baseMatrices; }


protected:
    CorotationalMeshMapping()
        : Inherit()
        , in_tetrahedra(initData(&in_tetrahedra,SeqTetrahedra(),"inputTetrahedra","input tetrahedra"))
        , in_hexahedra(initData(&in_hexahedra,SeqHexahedra(),"inputHexahedra","input hexahedra"))
//        , in_triangles(initData(&in_triangles,SeqTriangles(),"inputTriangles","input triangles"))
//        , in_quads(initData(&in_quads,SeqQuads(),"inputQuads","input quads"))
//        , in_edges(initData(&in_edges,SeqEdges(),"inputEdges","input edges"))
        , out_tetrahedra(initData(&out_tetrahedra,SeqTetrahedra(),"tetrahedra","output tetrahedra"))
        , out_hexahedra(initData(&out_hexahedra,SeqHexahedra(),"hexahedra","output hexahedra"))
//        , out_triangles(initData(&out_triangles,SeqTriangles(),"triangles","output triangles"))
//        , out_quads(initData(&out_quads,SeqQuads(),"quads","output quads"))
//        , out_edges(initData(&out_edges,SeqEdges(),"edges","output edges"))
    {
    }

    virtual ~CorotationalMeshMapping() {}

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    SparseKMatrixEigen geometricStiffness;               ///< Stiffness due to the non-linearity of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector

public:
    Data< SeqTetrahedra > in_tetrahedra;
    Data< SeqHexahedra > in_hexahedra;
//    Data< SeqTriangles > in_triangles;
//    Data< SeqQuads > in_quads;
//    Data< SeqEdges > in_edges;

    Data< SeqTetrahedra > out_tetrahedra;
    Data< SeqHexahedra > out_hexahedra;
//    Data< SeqTriangles > out_triangles;
//    Data< SeqQuads > out_quads;
//    Data< SeqEdges > out_edges;

//protected:
    VecVecID clusters;  ///< groups of points for which we compute the transformation
    VecVecID clusters_child;
    VecID index_childToParent;
    VecVecID index_parentToChild;
    helper::vector<Mat3x3> rot, rot0;
    VecCoord Xcm0;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CorotationalMeshMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API CorotationalMeshMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API CorotationalMeshMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
