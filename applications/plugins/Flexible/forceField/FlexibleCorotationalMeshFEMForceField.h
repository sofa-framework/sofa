#ifndef FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H
#define FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H


#include "../shapeFunction/BarycentricShapeFunction.h"
#include "../quadrature/TopologyGaussPointSampler.h"
#include "../deformationMapping/LinearMapping.h"
#include "../deformationMapping/CorotationalMeshMapping.h"
#include "../strainMapping/CauchyStrainJacobianBlock.h"
#include "../material/HookeForceField.h"

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseTopology/MeshTopology.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

/// WORK IN PROGRESS
/// meta-forcefield using Flexible components internally without adding extra computation neither extra memory
/// hard-coded implementation of corotational FEM
///
/// TODO
/// - optimization
///     -- work element by element rather than full corotationalDeformationMapping (or full with vectorialized SVD)
///     -- no need for full linearMapping (Jacobians should be enough since indices are easy to deduce)
///     -- preassemble constant sub-graph
/// - assembly API
/// - potential energy
/// - masks?
///
template<class DataTypes>
class SOFA_Flexible_API FlexibleCorotationalMeshFEMForceField : public core::behavior::ForceField<DataTypes>, public shapefunction::BarycentricShapeFunction<core::behavior::ShapeFunction>
{
public:


    SOFA_CLASS2(SOFA_TEMPLATE(FlexibleCorotationalMeshFEMForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField,DataTypes),SOFA_TEMPLATE(shapefunction::BarycentricShapeFunction,core::behavior::ShapeFunction));

    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName( const FlexibleCorotationalMeshFEMForceField<DataTypes>* = NULL) { return DataTypes::Name(); }

    /** @name  Input types    */
    //@{
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef core::behavior::MechanicalState<DataTypes> MStateType;
    //@}


    /** @name forceField functions */
    //@{
    virtual void init()
    {
        if( !this->mstate )
        {
            this->mstate = dynamic_cast<MStateType*>(this->getContext()->getMechanicalState());
            if( !this->mstate ) { serr<<"state not found"<< sendl; return; }
        }


        topology::MeshTopology *topo = NULL;
        this->getContext()->get( topo, core::objectmodel::BaseContext::SearchUp );
        if( !topo ) { serr<<"No MeshTopology found"<<sendl; return; }


/// ShapeFunction 1
        ShapeFunction::_state = this->mstate;
        ShapeFunction::parentTopology = topo;
        ShapeFunction::init();


/// CorotationalDeformationMapping
        m_corotationalDeformationMapping = core::objectmodel::New< CorotationalDeformationMapping >();
        m_corotationalDeformationMapping->setModels( this->mstate, m_rotatedDofs.get() );
//        m_corotationalDeformationMapping->in_edges.setParent( &topo->seqEdges );
//        m_corotationalDeformationMapping->in_triangles.setParent( &topo->seqTriangles );
//        m_corotationalDeformationMapping->in_quads.setParent( &topo->seqQuads );
        m_corotationalDeformationMapping->in_tetrahedra.setParent( &topo->seqTetrahedra );
        m_corotationalDeformationMapping->in_hexahedra.setParent( &topo->seqHexahedra );
        m_corotationalDeformationMapping->init();

/// Rotated Mesh
        m_rotatedTopology->seqPoints.setParent( &m_rotatedDofs->x );
//        m_rotatedTopology->seqEdges.setParent( &m_corotationalDeformationMapping->out_edges );
//        m_rotatedTopology->seqTriangles.setParent( &m_corotationalDeformationMapping->out_triangles );
//        m_rotatedTopology->seqQuads.setParent( &m_corotationalDeformationMapping->out_quads );
        m_rotatedTopology->seqTetrahedra.setParent( &m_corotationalDeformationMapping->out_tetrahedra );
        m_rotatedTopology->seqHexahedra.setParent( &m_corotationalDeformationMapping->out_hexahedra );

/// ShapeFunction 2
        m_internalShapeFunction->_state = m_rotatedDofs.get();
        m_internalShapeFunction->parentTopology = m_rotatedTopology.get();
        m_internalShapeFunction->init();

/// GaussPointSampler
        m_gaussPointSampler->f_method.setValue( 0 );
        m_gaussPointSampler->f_order.setValue( d_order.getValue() );
        m_gaussPointSampler->parentTopology = m_rotatedTopology.get();
        m_gaussPointSampler->f_inPosition.setParent( &m_rotatedDofs->x0 );
        m_gaussPointSampler->init();
        unsigned size = m_gaussPointSampler->getNbSamples();

/// Linear Mapping
//        _linearJacobianBlocks.resize( size );
        m_linearDeformationMapping = core::objectmodel::New< LinearDeformationMapping >( m_rotatedDofs.get(), m_deformationDofs.get() );
        m_linearDeformationMapping->_sampler = m_gaussPointSampler.get();
        m_linearDeformationMapping->_shapeFunction = m_internalShapeFunction.get();
        m_linearDeformationMapping->init();
        m_deformationDofs->resize(0); ///< was allocated by m_deformationMapping->init()...


/// Strain Mapping
        _strainJacobianBlocks.resize( size );


/// Material
        _materialBlocks.resize( size );
        for( unsigned int i=0 ; i<size ; i++ ) _materialBlocks[i].volume=&m_gaussPointSampler->f_volume.getValue()[i];



        ForceField::init();

        reinit();
    }

    virtual void reinit()
    {
        std::vector<Real> params; params.push_back( _youngModulus.getValue()); params.push_back(_poissonRatio.getValue());
        for( unsigned i=0; i < _materialBlocks.size() ; ++i )
        {
            _materialBlocks[i].init( params, _viscosity.getValue() );
        }

//        // reinit matrices
//        //if(this->assembleC.getValue()) updateC();
//        //if(this->assembleK.getValue()) updateK();
//        //if(this->assembleB.getValue()) updateB();

        ForceField::reinit();
        ShapeFunction::reinit();
    }




    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& _v)
    {
        m_corotationalDeformationMapping->apply( mparams, m_rotatedDofs->x ,_x);
        m_corotationalDeformationMapping->applyJ( mparams, m_rotatedDofs->v ,_v);

        const VecCoord& x = m_rotatedDofs->x.getValue();
        const VecDeriv& v = m_rotatedDofs->v.getValue();
        VecDeriv& f = *m_rotatedDofs->f.beginEdit();
        std::fill(f.begin(),f.end(),Deriv());


        typename LinearDeformationMapping::SparseMatrix& linearDeformationJacobianBlocks = m_linearDeformationMapping->getJacobianBlocks();


        // temporaries
        defaulttype::F331Types::Coord F;
        defaulttype::F331Types::Deriv VF;
        defaulttype::F331Types::Deriv PF;
        defaulttype::E331Types::Coord E;
        defaulttype::E331Types::Deriv VE;
        defaulttype::E331Types::Deriv PE;


        for( unsigned i=0; i < _strainJacobianBlocks.size() ; ++i )
        {
            F.clear();
            VF.clear();
            PF.clear();
            E.clear();
            VE.clear();
            PE.clear();

            for( unsigned int j=0 ; j<linearDeformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_linearDeformationMapping->f_index.getValue()[i][j];
                linearDeformationJacobianBlocks[i][j].addapply( F, x[index] );
                linearDeformationJacobianBlocks[i][j].addmult( VF, v[index] );
            }

            _strainJacobianBlocks[i].addapply( E, F );

            _strainJacobianBlocks[i].addmult( VE, VF );

            _materialBlocks[i].addForce( PE, E, VE );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            for( unsigned int j=0 ; j<linearDeformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_linearDeformationMapping->f_index.getValue()[i][j];
                linearDeformationJacobianBlocks[i][j].addMultTranspose( f[index], PF );
            }
        }

        m_rotatedDofs->f.endEdit();


        m_corotationalDeformationMapping->applyJT( mparams, _f, m_rotatedDofs->f );


//        /*if(!BlockType::constantK)
//        {
//            if(this->assembleC.getValue()) updateC();
//            if(this->assembleK.getValue()) updateK();
//            if(this->assembleB.getValue()) updateB();
//        }*/
    }

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  _df, const DataVecDeriv&  _dx )
    {
        m_corotationalDeformationMapping->applyJ( mparams, m_rotatedDofs->dx ,_dx);

        const VecDeriv& dx = m_rotatedDofs->dx.getValue();
        VecDeriv& df = *m_rotatedDofs->f.beginEdit();
        std::fill(df.begin(),df.end(),Deriv());

        typename LinearDeformationMapping::SparseMatrix& linearDeformationJacobianBlocks = m_linearDeformationMapping->getJacobianBlocks();

        // temporaries
        defaulttype::F331Types::Coord F;
        defaulttype::F331Types::Deriv PF;
        defaulttype::E331Types::Coord E;
        defaulttype::E331Types::Deriv PE;


        // TODO use masks
        for( unsigned i=0; i < _strainJacobianBlocks.size() ; ++i )
        {
            F.clear();
            PF.clear();
            E.clear();
            PE.clear();

            for( unsigned int j=0 ; j<linearDeformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_linearDeformationMapping->f_index.getValue()[i][j];
                linearDeformationJacobianBlocks[i][j].addmult( F, dx[index] );
            }

            _strainJacobianBlocks[i].addmult( E, F );

            _materialBlocks[i].addDForce( PE, E, mparams->kFactor(), mparams->bFactor() );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            for( unsigned int j=0 ; j<linearDeformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_linearDeformationMapping->f_index.getValue()[i][j];
                linearDeformationJacobianBlocks[i][j].addMultTranspose( df[index], PF );
            }
        }

        m_rotatedDofs->f.endEdit();

        m_corotationalDeformationMapping->applyJT( mparams, _df, m_rotatedDofs->f);
    }


    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
    {
        // TODO not implemented
        return 0;
    }

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }
    //@}


    typedef core::behavior::ForceField<DataTypes> ForceField;
    typedef shapefunction::BarycentricShapeFunction<core::behavior::ShapeFunction> ShapeFunction;
    typedef engine::TopologyGaussPointSampler GaussPointSampler;
    typedef mapping::LinearMapping< DataTypes, defaulttype::F331Types > LinearDeformationMapping;
    typedef mapping::CorotationalMeshMapping< DataTypes, DataTypes > CorotationalDeformationMapping;
    typedef topology::MeshTopology MeshTopology;

    typedef component::container::MechanicalObject < DataTypes > RotatedDofs;
    typedef component::container::MechanicalObject < defaulttype::F331Types > DeformationDofs;


    typedef defaulttype::LinearJacobianBlock< DataTypes, defaulttype::F331Types > LinearJacobianBlock;
    typedef vector< LinearJacobianBlock >  LinearJacobianBlocks;
//    LinearJacobianBlocks _linearJacobianBlocks;

    typedef defaulttype::CauchyStrainJacobianBlock< defaulttype::F331Types, defaulttype::E331Types > StrainJacobianBlock;
    typedef vector< StrainJacobianBlock >  StrainJacobianBlocks;
    StrainJacobianBlocks _strainJacobianBlocks;

	typedef defaulttype::IsotropicHookeLaw<typename defaulttype::E331Types::Real, defaulttype::E331Types::material_dimensions, defaulttype::E331Types::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock< defaulttype::E331Types, LawType > MaterialBlock;
    typedef vector< MaterialBlock >  MaterialBlocks;
    MaterialBlocks _materialBlocks;


    /** @name  Corotational methods */
    //@{
    enum DecompositionMethod { POLAR=0, QR, SMALL, SVD, NB_DecompositionMethod };
    Data<helper::OptionsGroup> d_method;
    //@}

    Data<unsigned> d_order; ///< order of spatial integration

    /** @name  Material parameters */
    //@{
    Data<Real> _youngModulus;
    Data<Real> _poissonRatio;
    Data<Real> _viscosity;
    //@}


    Data<bool> d_geometricStiffness; ///< should geometricStiffness be considered?



protected:

    typename LinearDeformationMapping::SPtr m_linearDeformationMapping;
    typename CorotationalDeformationMapping::SPtr m_corotationalDeformationMapping;
    typename RotatedDofs::SPtr m_rotatedDofs;
    DeformationDofs::SPtr m_deformationDofs;
    GaussPointSampler::SPtr m_gaussPointSampler;
    ShapeFunction::SPtr m_internalShapeFunction; // on rotated nodes
    MeshTopology::SPtr m_rotatedTopology;  // on rotated nodes


    FlexibleCorotationalMeshFEMForceField()
        : ForceField(), ShapeFunction()
        //, assembleC ( initData ( &assembleC,false, "assembleC","Assemble the Compliance matrix" ) )
        //, assembleK ( initData ( &assembleK,false, "assembleK","Assemble the Stiffness matrix" ) )
        //, assembleB ( initData ( &assembleB,false, "assembleB","Assemble the Damping matrix" ) )
        , d_method( initData( &d_method, "method", "Decomposition method" ) )
        , d_order( initData( &d_order, 1u, "order", "Order of quadrature method" ) )
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,(Real)0,"poissonRatio","Poisson Ratio"))
        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
        , d_geometricStiffness( initData( &d_geometricStiffness, false, "geometricStiffness", "Should geometricStiffness be considered?" ) )
    {
        helper::OptionsGroup Options;
        Options.setNbItems( NB_DecompositionMethod );
        Options.setItemName( SMALL, "small" );
        Options.setItemName( QR,    "qr"    );
        Options.setItemName( POLAR, "polar" );
        Options.setItemName( SVD,   "svd"   );
        Options.setSelectedItem( SVD );
        d_method.setValue( Options );

        m_rotatedDofs = core::objectmodel::New< RotatedDofs >(); m_rotatedDofs->x0.forceSet(); m_rotatedDofs->dx.forceSet();
        m_deformationDofs = core::objectmodel::New< DeformationDofs >();

        m_rotatedDofs.get()->forceMask.activate(false);
        m_deformationDofs.get()->forceMask.activate(false);

        m_gaussPointSampler = core::objectmodel::New< GaussPointSampler >();
        m_internalShapeFunction = core::objectmodel::New< ShapeFunction >();
        m_rotatedTopology = core::objectmodel::New< MeshTopology >();

    }

    virtual ~FlexibleCorotationalMeshFEMForceField() {}


    //Data<bool> assembleC;
    //SparseMatrixEigen C;

    /*  void updateC()
      {
          if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
          typename mstateType::ReadVecCoord X = this->mstate->readPositions();

          C.resizeBlocks(X.size(),X.size());
          for(unsigned int i=0;i<material.size();i++)
          {
              //        eigenJacobian.setBlock( i, i, jacobian[i].getJ());

              // Put all the blocks of the row in an array, then send the array to the matrix
              // Not very efficient: MatBlock creations could be avoided.
              vector<MatBlock> blocks;
              vector<unsigned> columns;
              columns.push_back( i );
              blocks.push_back( material[i].getC() );
              C.appendBlockRow( i, columns, blocks );
          }
          C.endEdit();
      }


      void updateK()
      {
          if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
          typename mstateType::ReadVecCoord X = this->mstate->readPositions();

          K.resizeBlocks(X.size(),X.size());
          for(unsigned int i=0;i<material.size();i++)
          {
              //        eigenJacobian.setBlock( i, i, jacobian[i].getJ());

              // Put all the blocks of the row in an array, then send the array to the matrix
              // Not very efficient: MatBlock creations could be avoided.
              vector<MatBlock> blocks;
              vector<unsigned> columns;
              columns.push_back( i );
              blocks.push_back( material[i].getK() );
              K.appendBlockRow( i, columns, blocks );
          }
          K.endEdit();
      }


      Data<bool> assembleB;
      SparseMatrixEigen B;

      void updateB()
      {
          if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
          typename mstateType::ReadVecCoord X = this->mstate->readPositions();

          B.resizeBlocks(X.size(),X.size());
          for(unsigned int i=0;i<material.size();i++)
          {
              //        eigenJacobian.setBlock( i, i, jacobian[i].getJ());

              // Put all the blocks of the row in an array, then send the array to the matrix
              // Not very efficient: MatBlock creations could be avoided.
              vector<MatBlock> blocks;
              vector<unsigned> columns;
              columns.push_back( i );
              blocks.push_back( material[i].getB() );
              B.appendBlockRow( i, columns, blocks );
          }
          B.endEdit();
      }*/




}; // class FlexibleCorotationalMeshFEMForceField



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_Flexible_API FlexibleCorotationalMeshFEMForceField<defaulttype::Vec3Types>;
#endif

} // namespace forcefield
} // namespace component
} // namespace sofa

#endif // FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H
