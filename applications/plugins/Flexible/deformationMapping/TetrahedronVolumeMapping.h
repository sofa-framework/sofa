#ifndef SOFA_COMPONENT_MAPPING_TetrahedronVolumeMapping_H
#define SOFA_COMPONENT_MAPPING_TetrahedronVolumeMapping_H

#include <Flexible/config.h>
#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>


namespace sofa
{
namespace component
{
namespace mapping
{



/** Map volumes from tetrahedra
 *
 * Two methods:
 *     - either a volume per tetra,
 *     - or volumes are dispatched per vertex (taking a quarter of its incident tetrahedra).
 *
 * @author Matthieu Nesme
 * @date 2014
 *
*/

template <class TIn, class TOut>
class TetrahedronVolumeMapping : public core::Mapping<TIn,TOut>
{
public:
    typedef core::Mapping<TIn,TOut> Inherit;
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
    typedef linearsolver::EigenSparseMatrix<TIn,TOut> SparseMatrixEigen;
    typedef linearsolver::EigenSparseMatrix<TIn,TIn> SparseKMatrixEigen;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
    typedef sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::helper::vector< Index > VecIndex;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

    SOFA_CLASS(SOFA_TEMPLATE2(TetrahedronVolumeMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    Data<OutVecCoord> d_offsets; ///< offsets removed from output volume

    /// dispatch the volume per nodes (each node taking a quarter of its incident tetrahedra)
    /// inspired from "Volume Conserving Finite Element Simulations of Deformable Models", Irving, Schroeder, Fedkiw, SIGGRAPH 2007
    Data<bool> d_volumePerNodes;

    virtual void init()
    {
        m_topology = this->getContext()->getMeshTopology();
        if( !m_topology ) { serr<<"No MeshTopology found."<<sendl; return; }
        int nbTetra = m_topology->getNbTetrahedra();
        if( !nbTetra ) { serr<<"The Topology constains no tetrahedra."<<sendl; return; }

        if( d_volumePerNodes.getValue() )
            this->getToModel()->resize( m_topology->getNbPoints() );
        else
            this->getToModel()->resize( nbTetra );

        baseMatrices.resize( 1 );
        baseMatrices[0] = &jacobian;

        this->Inherit::init();

        if( this->f_applyRestPosition.getValue() ) // copy rest pos as offsets
            d_offsets.setValue( this->toModel->read(core::ConstVecCoordId::position())->getValue() );

    }


    virtual void apply(const core::MechanicalParams*, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        if( !m_topology ) return;

        helper::WriteOnlyAccessor< Data<OutVecCoord> >  v = dOut;
        helper::ReadAccessor< Data<InVecCoord> >  x = dIn;

        jacobian.resizeBlocks( v.size(), x.size() );
        hessians.resize( m_topology->getNbTetrahedra() );

        typename Hessians::iterator hessianIt = hessians.begin();
        if( d_volumePerNodes.getValue() )
        {
            for( size_t i=0 ; i<v.size() ; i++ ) v[i][0] = 0;
            SparseMatrixEigen tetraJ( m_topology->getNbTetrahedra(), x.size()*Nin );
            SparseMatrixEigen TetraToNode( v.size(), m_topology->getNbTetrahedra() );
            for (int i = 0, nbTetra = m_topology->getNbTetrahedra(); i < nbTetra; i++)
            {
                Tetra t = m_topology->getTetra(i);
                Real tetravol = .25 * processTetra( i, t[0],t[1],t[2],t[3], x[t[0]],x[t[1]],x[t[2]],x[t[3]], tetraJ, *hessianIt );
                for( int j=0 ; j<4 ; ++j )
                {
                    TetraToNode.add( t[j], i, 0.25 );
                    v[t[j]][0] += tetravol;
                }
                ++hessianIt;
            }
            tetraJ.compress();
            TetraToNode.compress();
            jacobian.compressedMatrix = TetraToNode.compressedMatrix * tetraJ.compressedMatrix;
        }
        else
        {
            for (int i = 0, nbTetra = m_topology->getNbTetrahedra(); i < nbTetra; i++)
            {
                Tetra t = m_topology->getTetra(i);
                v[i][0] = processTetra( i, t[0],t[1],t[2],t[3], x[t[0]],x[t[1]],x[t[2]],x[t[3]], jacobian, *hessianIt );
            }
            ++hessianIt;
        }

        if( this->f_applyRestPosition.getValue() && !d_offsets.getValue().empty() )
        {
            const OutVecCoord& x0 = d_offsets.getValue();
            OutVecCoord& x = *dOut.beginWriteOnly();
            for( size_t i=0 ; i<x.size() ; ++i ) x[i] -= x0[std::min(i,x0.size()-1)];
            dOut.endEdit();
        }

        jacobian.compress();
    }

    virtual void applyJ(const core::MechanicalParams* /* mparams */, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)    { if( jacobian.rowSize() > 0 ) jacobian.mult(dOut,dIn);    }
    virtual void applyJT(const core::MechanicalParams* /* mparams */, Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)    { if( jacobian.rowSize() > 0 ) jacobian.addMultTranspose(dIn,dOut);    }
    virtual void applyJT(const core::ConstraintParams* /* cparams */, Data<InMatrixDeriv>& /*dIn*/, const Data<OutMatrixDeriv>& /*dOut*/) {}

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        geometricStiffness.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
    }


    virtual void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId )
    {
        size_t size = this->fromModel->getSize();
        geometricStiffness.resizeBlocks( size, size );

        const OutVecDeriv& childForce = childForceId[this->toModel.get(mparams)].read()->getValue();
        const OutVecDeriv* cf; // force per tetra
        if( d_volumePerNodes.getValue() )
        {
            // if per node -> compute force per tetra
            OutVecDeriv* localcf = new OutVecDeriv( m_topology->getNbTetrahedra() );
            cf = localcf;

            for (int i = 0, nbTetra = m_topology->getNbTetrahedra(); i < nbTetra; i++)
            {
                Tetra t = m_topology->getTetra(i);
                (*localcf)[i][0] += 0.25*(childForce[t[0]][0]+childForce[t[1]][0]+childForce[t[2]][0]+childForce[t[3]][0]);
            }
        }
        else cf = &childForce;

        typename Hessians::const_iterator hessianIt = hessians.begin();
        for( int i = 0, nbTetra = m_topology->getNbTetrahedra() ; i < nbTetra ; i++ )
        {
            Tetra t = m_topology->getTetra(i);

            for( int p0 = 0 ; p0 < 4 ; ++p0 )
            for( int p1 = 0 ; p1 < 4 ; ++p1 )
            for( int p0i = 0 ; p0i < Nin ; ++p0i )
            for( int p1i = 0 ; p1i < Nin ; ++p1i )
            {
                geometricStiffness.add( t[p0]*Nin+p0i, t[p1]*Nin+p1i, (*hessianIt)[p0*Nin+p0i][p1*Nin+p1i]*(*cf)[i][0] );
            }
            ++hessianIt;
            geometricStiffness.compress();
        }

        if( d_volumePerNodes.getValue() ) delete cf;
    }

    virtual const defaulttype::BaseMatrix* getK()
    {
        return &geometricStiffness;
    }

    virtual const sofa::defaulttype::BaseMatrix* getJ() { return &jacobian; }
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs()    { return &baseMatrices; }

protected:
    TetrahedronVolumeMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
        , d_offsets(initData(&d_offsets, "offsets", "offsets removed from output volume"))
        , d_volumePerNodes(initData(&d_volumePerNodes, "volumePerNodes", "Dispatch the volume on nodes"))
    {
    }

    virtual ~TetrahedronVolumeMapping()     { }


    sofa::core::topology::BaseMeshTopology* m_topology;  ///< where the triangles/quads are defined

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen geometricStiffness; ///< Stiffness due to the non-linearity of the mapping
    typedef defaulttype::Mat<12,12,Real> Hessian;
    typedef std::list<Hessian> Hessians;
    Hessians hessians; ///< local dJ per tetrahedron


    Real processTetra( int i,
                       const unsigned a, const unsigned b, const unsigned c, const unsigned d,
                       const InCoord A, const InCoord B, const InCoord C, const InCoord D,
                       SparseMatrixEigen& J, Hessian& H )
    {
        // TODO it can be optimized a lot!

        const Real& p00 = A[0];
        const Real& p01 = A[1];
        const Real& p02 = A[2];
        const Real& p10 = B[0];
        const Real& p11 = B[1];
        const Real& p12 = B[2];
        const Real& p20 = C[0];
        const Real& p21 = C[1];
        const Real& p22 = C[2];
        const Real& p30 = D[0];
        const Real& p31 = D[1];
        const Real& p32 = D[2];

        static const Real inv6 = 1./6.;

        Real t1 = p11 * p22;
        Real t3 = p11 * p32;
        Real t5 = p12 * p21;
        Real t7 = p12 * p31;
        Real t9 = p21 * p32;
        Real t11 = p22 * p31;
        Real t13 = p10 * p22;
        Real t15 = p01 * p10;
        Real t17 = p12 * p20;
        Real t19 = p01 * p12;
        Real t21 = p01 * p20;
        Real t23 = p01 * p22;
        Real t25 = p00 * t1 - p00 * t11 - p00 * t3 - p00 * t5 + p00 * t7 + p00 * t9 - p01 * t13 + p01 * t17 - p30 * t19 + p30 * t23 + p32 * t15 - p32 * t21;
        Real t26 = p10 * p21;
        Real t28 = p02 * p10;
        Real t30 = p11 * p20;
        Real t32 = p02 * p11;
        Real t34 = p02 * p20;
        Real t36 = p02 * p21;
        Real t44 = p02 * t26 - p02 * t30 - p30 * t1 + p30 * t32 - p30 * t36 + p30 * t5 + p31 * t13 - p31 * t17 - p31 * t28 + p31 * t34 - p32 * t26 + p32 * t30;
        Real t45 = t25 + t44;
        Real t46 = t45 * t45;
        Real t47 = sqrt(t46);
        Real t50 = 0.1e1 / t47;
        Real t51 = t1 - t3 - t5 + t7 + t9 - t11;
        Real t52 = t51 * t51;
        Real t54 = 0.1e1 / t47;
        Real t57 = p10 * p32;
        Real t58 = p12 * p30;
        Real t59 = p20 * p32;
        Real t60 = p22 * p30;
        Real t61 = -t13 + t57 + t17 - t58 - t59 + t60;
        Real t66 = -t61 * t51 * t50 + t51 * t61 * t54;
        Real t67 = p10 * p31;
        Real t68 = p11 * p30;
        Real t69 = p20 * p31;
        Real t70 = p21 * p30;
        Real t71 = t26 - t67 - t30 + t68 + t69 - t70;
        Real t74 = t71 * t54;
        Real t76 = -t71 * t51 * t50 + t51 * t74;
        Real t77 = p01 * p32;
        Real t78 = p02 * p31;
        Real t79 = -t23 + t77 + t36 - t78 - t9 + t11;
        Real t82 = t79 * t54;
        Real t84 = -t79 * t51 * t50 + t51 * t82;
        Real t85 = p00 * p22;
        Real t86 = p00 * p32;
        Real t87 = p02 * p30;
        Real t88 = t85 - t86 - t34 + t87 + t59 - t60;
        Real t91 = t88 * t54;
        Real t93 = t45 * t54;
        Real t94 = p22 - p32;
        Real t96 = -t88 * t51 * t50 + t51 * t91 + t94 * t93;
        Real t97 = p00 * p21;
        Real t98 = p00 * p31;
        Real t99 = p01 * p30;
        Real t100 = -t97 + t98 + t21 - t99 - t69 + t70;
        Real t103 = t100 * t54;
        Real t105 = p31 - p21;
        Real t107 = -t100 * t51 * t50 + t51 * t103 + t105 * t93;
        Real t108 = t19 - t77 - t32 + t78 + t3 - t7;
        Real t111 = t108 * t54;
        Real t113 = -t108 * t51 * t50 + t51 * t111;
        Real t114 = p00 * p12;
        Real t115 = -t114 + t86 + t28 - t87 - t57 + t58;
        Real t118 = t115 * t54;
        Real t120 = p32 - p12;
        Real t122 = -t115 * t51 * t50 + t51 * t118 + t120 * t93;
        Real t123 = p00 * p11;
        Real t124 = t123 - t98 - t15 + t99 + t67 - t68;
        Real t127 = t124 * t54;
        Real t129 = p11 - p31;
        Real t131 = -t124 * t51 * t50 + t51 * t127 + t129 * t93;
        Real t132 = -t19 + t23 + t32 - t36 - t1 + t5;
        Real t135 = t132 * t54;
        Real t137 = -t132 * t51 * t50 + t51 * t135;
        Real t138 = t114 - t85 - t28 + t34 + t13 - t17;
        Real t141 = t138 * t54;
        Real t143 = p12 - p22;
        Real t145 = -t138 * t51 * t50 + t51 * t141 + t143 * t93;
        Real t146 = -t123 + t97 + t15 - t21 - t26 + t30;
        Real t149 = t146 * t54;
        Real t151 = p21 - p11;
        Real t153 = -t146 * t51 * t50 + t51 * t149 + t151 * t93;
        Real t154 = t61 * t61;
        Real t161 = -t71 * t61 * t50 + t61 * t74;
        Real t166 = -t79 * t61 * t50 + t61 * t82 - t94 * t93;
        Real t170 = -t88 * t61 * t50 + t61 * t91;
        Real t174 = p20 - p30;
        Real t176 = -t100 * t61 * t50 + t61 * t103 + t174 * t93;
        Real t181 = -t108 * t61 * t50 + t61 * t111 - t120 * t93;
        Real t185 = -t115 * t61 * t50 + t61 * t118;
        Real t189 = p30 - p10;
        Real t191 = -t124 * t61 * t50 + t61 * t127 + t189 * t93;
        Real t196 = -t132 * t61 * t50 + t61 * t135 - t143 * t93;
        Real t200 = -t138 * t61 * t50 + t61 * t141;
        Real t204 = p10 - p20;
        Real t206 = -t146 * t61 * t50 + t61 * t149 + t204 * t93;
        Real t207 = t71 * t71;
        Real t215 = -t79 * t71 * t50 - t105 * t93 + t71 * t82;
        Real t220 = -t88 * t71 * t50 - t174 * t93 + t71 * t91;
        Real t224 = -t100 * t71 * t50 + t71 * t103;
        Real t229 = -t108 * t71 * t50 + t71 * t111 - t129 * t93;
        Real t234 = -t115 * t71 * t50 + t71 * t118 - t189 * t93;
        Real t238 = -t124 * t71 * t50 + t71 * t127;
        Real t243 = -t132 * t71 * t50 + t71 * t135 - t151 * t93;
        Real t248 = -t138 * t71 * t50 + t71 * t141 - t204 * t93;
        Real t252 = -t146 * t71 * t50 + t71 * t149;
        Real t253 = t79 * t79;
        Real t260 = -t88 * t79 * t50 + t79 * t91;
        Real t264 = -t100 * t79 * t50 + t79 * t103;
        Real t268 = -t108 * t79 * t50 + t79 * t111;
        Real t272 = p02 - p32;
        Real t274 = -t115 * t79 * t50 + t79 * t118 + t272 * t93;
        Real t278 = p31 - p01;
        Real t280 = -t124 * t79 * t50 + t79 * t127 + t278 * t93;
        Real t284 = -t132 * t79 * t50 + t79 * t135;
        Real t288 = p22 - p02;
        Real t290 = -t138 * t79 * t50 + t79 * t141 + t288 * t93;
        Real t294 = p01 - p21;
        Real t296 = -t146 * t79 * t50 + t79 * t149 + t294 * t93;
        Real t297 = t88 * t88;
        Real t304 = -t100 * t88 * t50 + t88 * t103;
        Real t309 = -t108 * t88 * t50 + t88 * t111 - t272 * t93;
        Real t313 = -t115 * t88 * t50 + t88 * t118;
        Real t317 = p00 - p30;
        Real t319 = -t124 * t88 * t50 + t88 * t127 + t317 * t93;
        Real t324 = -t132 * t88 * t50 + t88 * t135 - t288 * t93;
        Real t328 = -t138 * t88 * t50 + t88 * t141;
        Real t332 = p20 - p00;
        Real t334 = -t146 * t88 * t50 + t88 * t149 + t332 * t93;
        Real t335 = t100 * t100;
        Real t343 = -t108 * t100 * t50 + t100 * t111 - t278 * t93;
        Real t348 = -t115 * t100 * t50 + t100 * t118 - t317 * t93;
        Real t352 = -t124 * t100 * t50 + t100 * t127;
        Real t357 = -t132 * t100 * t50 + t100 * t135 - t294 * t93;
        Real t362 = -t138 * t100 * t50 + t100 * t141 - t332 * t93;
        Real t366 = -t146 * t100 * t50 + t100 * t149;
        Real t367 = t108 * t108;
        Real t374 = -t115 * t108 * t50 + t108 * t118;
        Real t378 = -t124 * t108 * t50 + t108 * t127;
        Real t382 = -t132 * t108 * t50 + t108 * t135;
        Real t386 = p02 - p12;
        Real t388 = -t138 * t108 * t50 + t108 * t141 + t386 * t93;
        Real t392 = p11 - p01;
        Real t394 = -t146 * t108 * t50 + t108 * t149 + t392 * t93;
        Real t395 = t115 * t115;
        Real t402 = -t124 * t115 * t50 + t115 * t127;
        Real t407 = -t132 * t115 * t50 + t115 * t135 - t386 * t93;
        Real t411 = -t138 * t115 * t50 + t115 * t141;
        Real t415 = p00 - p10;
        Real t417 = -t146 * t115 * t50 + t115 * t149 + t415 * t93;
        Real t418 = t124 * t124;
        Real t426 = -t132 * t124 * t50 + t124 * t135 - t392 * t93;
        Real t431 = -t138 * t124 * t50 + t124 * t141 - t415 * t93;
        Real t435 = -t146 * t124 * t50 + t124 * t149;
        Real t436 = t132 * t132;
        Real t443 = -t138 * t132 * t50 + t132 * t141;
        Real t447 = -t146 * t132 * t50 + t132 * t149;
        Real t448 = t138 * t138;
        Real t455 = -t146 * t138 * t50 + t138 * t149;
        Real t456 = t146 * t146;
        Real t462 = t51 * t93 * inv6;
        Real t464 = t61 * t93 * inv6;
        Real t466 = t71 * t93 * inv6;
        Real t468 = t79 * t93 * inv6;
        Real t470 = t88 * t93 * inv6;
        Real t472 = t100 * t93 * inv6;
        Real t474 = t108 * t93 * inv6;
        Real t476 = t115 * t93 * inv6;
        Real t478 = t124 * t93 * inv6;
        Real t480 = t132 * t93 * inv6;
        Real t482 = t138 * t93 * inv6;

        // K
        H[0*Nin+0][0*Nin+0] = -t52 * t50 * inv6 + t52 * t54 * inv6 ;
        H[0*Nin+0][0*Nin+1] = t66 * inv6 ;
        H[0*Nin+0][0*Nin+2] = t76 * inv6 ;
        H[0*Nin+0][1*Nin+0] = t84 * inv6 ;
        H[0*Nin+0][1*Nin+1] = t96 * inv6 ;
        H[0*Nin+0][1*Nin+2] = t107 * inv6 ;
        H[0*Nin+0][2*Nin+0] = t113 * inv6 ;
        H[0*Nin+0][2*Nin+1] = t122 * inv6 ;
        H[0*Nin+0][2*Nin+2] = t131 * inv6 ;
        H[0*Nin+0][3*Nin+0] = t137 * inv6 ;
        H[0*Nin+0][3*Nin+1] = t145 * inv6 ;
        H[0*Nin+0][3*Nin+2] = t153 * inv6 ;
        H[0*Nin+1][0*Nin+0] = t66 * inv6 ;
        H[0*Nin+1][0*Nin+1] = -t154 * t50 * inv6 + t154 * t54 * inv6 ;
        H[0*Nin+1][0*Nin+2] = t161 * inv6 ;
        H[0*Nin+1][1*Nin+0] = t166 * inv6 ;
        H[0*Nin+1][1*Nin+1] = t170 * inv6 ;
        H[0*Nin+1][1*Nin+2] = t176 * inv6 ;
        H[0*Nin+1][2*Nin+0] = t181 * inv6 ;
        H[0*Nin+1][2*Nin+1] = t185 * inv6 ;
        H[0*Nin+1][2*Nin+2] = t191 * inv6 ;
        H[0*Nin+1][3*Nin+0] = t196 * inv6 ;
        H[0*Nin+1][3*Nin+1] = t200 * inv6 ;
        H[0*Nin+1][3*Nin+2] = t206 * inv6 ;
        H[0*Nin+2][0*Nin+0] = t76 * inv6 ;
        H[0*Nin+2][0*Nin+1] = t161 * inv6 ;
        H[0*Nin+2][0*Nin+2] = -t207 * t50 * inv6 + t207 * t54 * inv6 ;
        H[0*Nin+2][1*Nin+0] = t215 * inv6 ;
        H[0*Nin+2][1*Nin+1] = t220 * inv6 ;
        H[0*Nin+2][1*Nin+2] = t224 * inv6 ;
        H[0*Nin+2][2*Nin+0] = t229 * inv6 ;
        H[0*Nin+2][2*Nin+1] = t234 * inv6 ;
        H[0*Nin+2][2*Nin+2] = t238 * inv6 ;
        H[0*Nin+2][3*Nin+0] = t243 * inv6 ;
        H[0*Nin+2][3*Nin+1] = t248 * inv6 ;
        H[0*Nin+2][3*Nin+2] = t252 * inv6 ;
        H[1*Nin+0][0*Nin+0] = t84 * inv6 ;
        H[1*Nin+0][0*Nin+1] = t166 * inv6 ;
        H[1*Nin+0][0*Nin+2] = t215 * inv6 ;
        H[1*Nin+0][1*Nin+0] = -t253 * t50 * inv6 + t253 * t54 * inv6 ;
        H[1*Nin+0][1*Nin+1] = t260 * inv6 ;
        H[1*Nin+0][1*Nin+2] = t264 * inv6 ;
        H[1*Nin+0][2*Nin+0] = t268 * inv6 ;
        H[1*Nin+0][2*Nin+1] = t274 * inv6 ;
        H[1*Nin+0][2*Nin+2] = t280 * inv6 ;
        H[1*Nin+0][3*Nin+0] = t284 * inv6 ;
        H[1*Nin+0][3*Nin+1] = t290 * inv6 ;
        H[1*Nin+0][3*Nin+2] = t296 * inv6 ;
        H[1*Nin+1][0*Nin+0] = t96 * inv6 ;
        H[1*Nin+1][0*Nin+1] = t170 * inv6 ;
        H[1*Nin+1][0*Nin+2] = t220 * inv6 ;
        H[1*Nin+1][1*Nin+0] = t260 * inv6 ;
        H[1*Nin+1][1*Nin+1] = -t297 * t50 * inv6 + t297 * t54 * inv6 ;
        H[1*Nin+1][1*Nin+2] = t304 * inv6 ;
        H[1*Nin+1][2*Nin+0] = t309 * inv6 ;
        H[1*Nin+1][2*Nin+1] = t313 * inv6 ;
        H[1*Nin+1][2*Nin+2] = t319 * inv6 ;
        H[1*Nin+1][3*Nin+0] = t324 * inv6 ;
        H[1*Nin+1][3*Nin+1] = t328 * inv6 ;
        H[1*Nin+1][3*Nin+2] = t334 * inv6 ;
        H[1*Nin+2][0*Nin+0] = t107 * inv6 ;
        H[1*Nin+2][0*Nin+1] = t176 * inv6 ;
        H[1*Nin+2][0*Nin+2] = t224 * inv6 ;
        H[1*Nin+2][1*Nin+0] = t264 * inv6 ;
        H[1*Nin+2][1*Nin+1] = t304 * inv6 ;
        H[1*Nin+2][1*Nin+2] = -t335 * t50 * inv6 + t335 * t54 * inv6 ;
        H[1*Nin+2][2*Nin+0] = t343 * inv6 ;
        H[1*Nin+2][2*Nin+1] = t348 * inv6 ;
        H[1*Nin+2][2*Nin+2] = t352 * inv6 ;
        H[1*Nin+2][3*Nin+0] = t357 * inv6 ;
        H[1*Nin+2][3*Nin+1] = t362 * inv6 ;
        H[1*Nin+2][3*Nin+2] = t366 * inv6 ;
        H[2*Nin+0][0*Nin+0] = t113 * inv6 ;
        H[2*Nin+0][0*Nin+1] = t181 * inv6 ;
        H[2*Nin+0][0*Nin+2] = t229 * inv6 ;
        H[2*Nin+0][1*Nin+0] = t268 * inv6 ;
        H[2*Nin+0][1*Nin+1] = t309 * inv6 ;
        H[2*Nin+0][1*Nin+2] = t343 * inv6 ;
        H[2*Nin+0][2*Nin+0] = -t367 * t50 * inv6 + t367 * t54 * inv6 ;
        H[2*Nin+0][2*Nin+1] = t374 * inv6 ;
        H[2*Nin+0][2*Nin+2] = t378 * inv6 ;
        H[2*Nin+0][3*Nin+0] = t382 * inv6 ;
        H[2*Nin+0][3*Nin+1] = t388 * inv6 ;
        H[2*Nin+0][3*Nin+2] = t394 * inv6 ;
        H[2*Nin+1][0*Nin+0] = t122 * inv6 ;
        H[2*Nin+1][0*Nin+1] = t185 * inv6 ;
        H[2*Nin+1][0*Nin+2] = t234 * inv6 ;
        H[2*Nin+1][1*Nin+0] = t274 * inv6 ;
        H[2*Nin+1][1*Nin+1] = t313 * inv6 ;
        H[2*Nin+1][1*Nin+2] = t348 * inv6 ;
        H[2*Nin+1][2*Nin+0] = t374 * inv6 ;
        H[2*Nin+1][2*Nin+1] = -t395 * t50 * inv6 + t395 * t54 * inv6 ;
        H[2*Nin+1][2*Nin+2] = t402 * inv6 ;
        H[2*Nin+1][3*Nin+0] = t407 * inv6 ;
        H[2*Nin+1][3*Nin+1] = t411 * inv6 ;
        H[2*Nin+1][3*Nin+2] = t417 * inv6 ;
        H[2*Nin+2][0*Nin+0] = t131 * inv6 ;
        H[2*Nin+2][0*Nin+1] = t191 * inv6 ;
        H[2*Nin+2][0*Nin+2] = t238 * inv6 ;
        H[2*Nin+2][1*Nin+0] = t280 * inv6 ;
        H[2*Nin+2][1*Nin+1] = t319 * inv6 ;
        H[2*Nin+2][1*Nin+2] = t352 * inv6 ;
        H[2*Nin+2][2*Nin+0] = t378 * inv6 ;
        H[2*Nin+2][2*Nin+1] = t402 * inv6 ;
        H[2*Nin+2][2*Nin+2] = -t418 * t50 * inv6 + t418 * t54 * inv6 ;
        H[2*Nin+2][3*Nin+0] = t426 * inv6 ;
        H[2*Nin+2][3*Nin+1] = t431 * inv6 ;
        H[2*Nin+2][3*Nin+2] = t435 * inv6 ;
        H[3*Nin+0][0*Nin+0] = t137 * inv6 ;
        H[3*Nin+0][0*Nin+1] = t196 * inv6 ;
        H[3*Nin+0][0*Nin+2] = t243 * inv6 ;
        H[3*Nin+0][1*Nin+0] = t284 * inv6 ;
        H[3*Nin+0][1*Nin+1] = t324 * inv6 ;
        H[3*Nin+0][1*Nin+2] = t357 * inv6 ;
        H[3*Nin+0][2*Nin+0] = t382 * inv6 ;
        H[3*Nin+0][2*Nin+1] = t407 * inv6 ;
        H[3*Nin+0][2*Nin+2] = t426 * inv6 ;
        H[3*Nin+0][3*Nin+0] = -t436 * t50 * inv6 + t436 * t54 * inv6 ;
        H[3*Nin+0][3*Nin+1] = t443 * inv6 ;
        H[3*Nin+0][3*Nin+2] = t447 * inv6 ;
        H[3*Nin+1][0*Nin+0] = t145 * inv6 ;
        H[3*Nin+1][0*Nin+1] = t200 * inv6 ;
        H[3*Nin+1][0*Nin+2] = t248 * inv6 ;
        H[3*Nin+1][1*Nin+0] = t290 * inv6 ;
        H[3*Nin+1][1*Nin+1] = t328 * inv6 ;
        H[3*Nin+1][1*Nin+2] = t362 * inv6 ;
        H[3*Nin+1][2*Nin+0] = t388 * inv6 ;
        H[3*Nin+1][2*Nin+1] = t411 * inv6 ;
        H[3*Nin+1][2*Nin+2] = t431 * inv6 ;
        H[3*Nin+1][3*Nin+0] = t443 * inv6 ;
        H[3*Nin+1][3*Nin+1] = -t448 * t50 * inv6 + t448 * t54 * inv6 ;
        H[3*Nin+1][3*Nin+2] = t455 * inv6 ;
        H[3*Nin+2][0*Nin+0] = t153 * inv6 ;
        H[3*Nin+2][0*Nin+1] = t206 * inv6 ;
        H[3*Nin+2][0*Nin+2] = t252 * inv6 ;
        H[3*Nin+2][1*Nin+0] = t296 * inv6 ;
        H[3*Nin+2][1*Nin+1] = t334 * inv6 ;
        H[3*Nin+2][1*Nin+2] = t366 * inv6 ;
        H[3*Nin+2][2*Nin+0] = t394 * inv6 ;
        H[3*Nin+2][2*Nin+1] = t417 * inv6 ;
        H[3*Nin+2][2*Nin+2] = t435 * inv6 ;
        H[3*Nin+2][3*Nin+0] = t447 * inv6 ;
        H[3*Nin+2][3*Nin+1] = t455 * inv6 ;
        H[3*Nin+2][3*Nin+2] = -t456 * t50 * inv6 + t456 * t54 * inv6 ;


        // J
        J.add(i, a*Nin+0,t462 );
        J.add(i, a*Nin+1,t464 );
        J.add(i, a*Nin+2,t466 );
        J.add(i, b*Nin+0,t468 );
        J.add(i, b*Nin+1,t470 );
        J.add(i, b*Nin+2,t472 );
        J.add(i, c*Nin+0,t474 );
        J.add(i, c*Nin+1,t476 );
        J.add(i, c*Nin+2,t478 );
        J.add(i, d*Nin+0,t480 );
        J.add(i, d*Nin+1, t482 );
        J.add(i, d*Nin+2, t146 * t93 * inv6 );

        // tetra volume
        return  t47 * inv6;
    }
};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
