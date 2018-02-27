#ifndef FLEXIBLE_METAFEMFORCEFIELD_H
#define FLEXIBLE_METAFEMFORCEFIELD_H


#include "../shapeFunction/BarycentricShapeFunction.h"
#include "../quadrature/TopologyGaussPointSampler.h"
#include "../deformationMapping/LinearMapping.h"
#include "../strainMapping/CorotationalStrainMapping.h"
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
/// - assembly
/// - potential energy
/// - masks?
/// - more generic (templated on strain mapping and material)
///
/// @author Matthieu Nesme
///
template<class DataTypes>
class SOFA_Flexible_API FlexibleCorotationalFEMForceField : public core::behavior::ForceField<DataTypes>, public shapefunction::BarycentricShapeFunction<core::behavior::ShapeFunction>
{
public:


    SOFA_CLASS2(SOFA_TEMPLATE(FlexibleCorotationalFEMForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField,DataTypes),SOFA_TEMPLATE(shapefunction::BarycentricShapeFunction,core::behavior::ShapeFunction));

    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName( const FlexibleCorotationalFEMForceField<DataTypes>* = NULL) { return DataTypes::Name(); }

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


/// ShapeFunction
        ShapeFunction::_state = this->mstate;
        ShapeFunction::parentTopology = topo;
        ShapeFunction::init();


/// GaussPointSampler
        m_gaussPointSampler->f_method.setValue( 0 );
        m_gaussPointSampler->f_order.setValue( d_order.getValue() );
        m_gaussPointSampler->parentTopology = topo;
        m_gaussPointSampler->f_inPosition.setParent( &topo->seqPoints );
        m_gaussPointSampler->init();


/// DeformationMapping
        m_deformationMapping = core::objectmodel::New< DeformationMapping >( this->mstate, m_deformationDofs.get() );
        m_deformationMapping->_sampler = m_gaussPointSampler.get();
        m_deformationMapping->_shapeFunction = (ShapeFunction*)this;
        m_deformationMapping->init();
        m_deformationDofs->resize(0); ///< was allocated by m_deformationMapping->init()...

        unsigned size = m_gaussPointSampler->getNbSamples();




/// Strain Mapping
        _strainJacobianBlocks.resize( size );


/// Material
        _materialBlocks.resize( size );
        for( unsigned int i=0 ; i<size ; i++ ) _materialBlocks[i].volume=&m_gaussPointSampler->f_volume.getValue()[i];



        ForceField::init();

        reinit();



        if( d_geometricStiffness.getValue() )
        {
             _stresses.resize(size);
        }

    }

    virtual void reinit()
    {
//        _lambda = _youngModulus.getValue()*_poissonRatio.getValue()/((1-2*_poissonRatio.getValue())*(1+_poissonRatio.getValue()));
//        _mu2    = _youngModulus.getValue()/(1+_poissonRatio.getValue());

        switch( d_method.getValue().getSelectedId() )
        {
            case SMALL:
            {
                for( size_t i=0 ; i<_strainJacobianBlocks.size() ; i++ )
                {
                    _strainJacobianBlocks[i].init_small();
                }
                break;
            }
            case QR:
            {
                for( size_t i=0 ; i<_strainJacobianBlocks.size() ; i++ )
                {
                    _strainJacobianBlocks[i].init_qr( d_geometricStiffness.getValue() );
                }
                break;
            }
            case POLAR:
            {
                for( size_t i=0 ; i<_strainJacobianBlocks.size() ; i++ )
                {
                    _strainJacobianBlocks[i].init_polar( d_geometricStiffness.getValue() );
                }
                break;
            }
            case SVD:
            {
                for( size_t i=0 ; i<_strainJacobianBlocks.size() ; i++ )
                {
                    _strainJacobianBlocks[i].init_svd( d_geometricStiffness.getValue() );
                }
                break;
            }
        }

        std::vector<Real> params; params.push_back( _youngModulus.getValue()); params.push_back(_poissonRatio.getValue());
        for( unsigned i=0; i < _materialBlocks.size() ; ++i )
        {
            _materialBlocks[i].init( params, _viscosity.getValue() );
        }

        // reinit matrices
        //if(this->assembleC.getValue()) updateC();
        //if(this->assembleK.getValue()) updateK();
        //if(this->assembleB.getValue()) updateB();

        ForceField::reinit();
        ShapeFunction::reinit();
    }



    virtual void addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& _v)
    {
        VecDeriv& f = *_f.beginEdit();
        const VecCoord& x = _x.getValue();
        const VecDeriv& v = _v.getValue();

        typename DeformationMapping::SparseMatrix& deformationJacobianBlocks = m_deformationMapping->getJacobianBlocks();

        DecompositionMethod method = (DecompositionMethod)d_method.getValue().getSelectedId();
        bool geometricStiffness = d_geometricStiffness.getValue();


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

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addapply( F, x[index] );
                deformationJacobianBlocks[i][j].addmult( VF, v[index] );
            }

            // TODO move the switch outside of the loop
            switch( method )
            {
                case SMALL:
                {
                    _strainJacobianBlocks[i].addapply_small( E, F );
                    break;
                }
                case QR:
                {
                    _strainJacobianBlocks[i].addapply_qr( E, F );
                    break;
                }
                case POLAR:
                {
                    _strainJacobianBlocks[i].addapply_polar( E, F );
                    break;
                }
                case SVD:
                {
                    _strainJacobianBlocks[i].addapply_svd( E, F );
                    break;
                }
                default:
                    break;
            }

            _strainJacobianBlocks[i].addmult( VE, VF );

            _materialBlocks[i].addForce( PE, E, VE );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addMultTranspose( f[index], PF );
            }


            if( geometricStiffness )
            {
                _stresses[i] = PE;
            }
        }

        _f.endEdit();

        /*if(!BlockType::constantK)
        {
            if(this->assembleC.getValue()) updateC();
            if(this->assembleK.getValue()) updateK();
            if(this->assembleB.getValue()) updateB();
        }*/
    }





    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& _df, const DataVecDeriv& _dx )
    {
        VecDeriv& df = *_df.beginEdit();
        const VecDeriv& dx = _dx.getValue();


        typename DeformationMapping::SparseMatrix& deformationJacobianBlocks = m_deformationMapping->getJacobianBlocks();


        DecompositionMethod method = (DecompositionMethod)d_method.getValue().getSelectedId();
        bool geometricStiffness = d_geometricStiffness.getValue();

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

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addmult( F, dx[index] );
            }

            _strainJacobianBlocks[i].addmult( E, F );

            _materialBlocks[i].addDForce( PE, E, mparams->kFactor(), mparams->bFactor() );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            // geometricStiffness: need for saving stress and deformation gradients
            if( geometricStiffness )
            {
                // TODO move the switch outside of the loop
                switch( method )
                {
                    case QR:
                    {
                        _strainJacobianBlocks[i].addDForce_qr( PF, F, _stresses[i], mparams->kFactor() );
                        break;
                    }
                    case POLAR:
                    {
                        _strainJacobianBlocks[i].addDForce_polar( PF, F, _stresses[i], mparams->kFactor() );
                        break;
                    }
                    case SVD:
                    {
                        _strainJacobianBlocks[i].addDForce_svd( PF, F, _stresses[i], mparams->kFactor() );
                        break;
                    }
                    default:
                        break;
                }
            }

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = m_deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addMultTranspose( df[index], PF );
            }
        }

        _df.endEdit();

    }


//    const defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams */*mparams*/)
//    {
//        if( !isCompliance.getValue() ) return NULL; // if seen as a stiffness, then return no compliance matrix
//        if(!this->assembleC.getValue()) updateC();
//        return &C;
//    }

//    virtual const sofa::defaulttype::BaseMatrix* getStiffnessMatrix(const core::MechanicalParams*)
//    {
//        if( isCompliance.getValue() ) return NULL; // if seen as a compliance, then return no stiffness matrix
//        if(!this->assembleK.getValue()) updateK();
////        cerr<<"BaseMaterialForceField::getStiffnessMatrix, K = " << K << endl;
//        return &K;
//    }

//    const defaulttype::BaseMatrix* getB(const core::MechanicalParams */*mparams*/)
//    {
//        if(!this->assembleB.getValue()) updateB();
//        return &B;
//    }



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
    typedef mapping::LinearMapping< DataTypes, defaulttype::F331Types > DeformationMapping;
    //typedef mapping::CorotationalStrainMapping< defaulttype::F331Types, defaulttype::E331Types > StrainMapping;
    //typedef HookeForceField< defaulttype::E331Types > MaterialForceField;

    typedef component::container::MechanicalObject < defaulttype::F331Types > DeformationDofs;
    //typedef component::container::MechanicalObject < defaulttype::E331Types > StrainDofs;

    //typedef defaulttype::LinearJacobianBlock< defaulttype::F331Types, defaulttype::E331Types > DeformationJacobianBlock;
    //typedef helper::vector< DeformationJacobianBlock >  DeformationJacobianBlocks;
    //DeformationJacobianBlocks _deformationJacobianBlocks;

    typedef defaulttype::CorotationalStrainJacobianBlock< defaulttype::F331Types, defaulttype::E331Types > StrainJacobianBlock;
    typedef helper::vector< StrainJacobianBlock >  StrainJacobianBlocks;
    StrainJacobianBlocks _strainJacobianBlocks;

	typedef defaulttype::IsotropicHookeLaw<typename defaulttype::E331Types::Real, defaulttype::E331Types::material_dimensions, defaulttype::E331Types::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock< defaulttype::E331Types, LawType > MaterialBlock;
    typedef helper::vector< MaterialBlock >  MaterialBlocks;
    MaterialBlocks _materialBlocks;


    /** @name  Corotational methods */
    //@{
    enum DecompositionMethod { POLAR=0, QR, SMALL, SVD, NB_DecompositionMethod };
    Data<helper::OptionsGroup> d_method; ///< Decomposition method
    //@}

    Data<unsigned> d_order; ///< order of spatial integration

    /** @name  Material parameters */
    //@{
    Data<Real> _youngModulus; ///< Young Modulus
    Data<Real> _poissonRatio; ///< Poisson Ratio
    Data<Real> _viscosity; ///< Viscosity (stress/strainRate)
//    Real _lambda;  ///< Lamé first coef
//    Real _mu2;     ///< Lamé second coef * 2
    //@}


    Data<bool> d_geometricStiffness; ///< should geometricStiffness be considered?



protected:


    typename DeformationMapping::SPtr m_deformationMapping;
    DeformationDofs::SPtr m_deformationDofs;
    GaussPointSampler::SPtr m_gaussPointSampler;
    defaulttype::E331Types::VecDeriv _stresses; // optionaln for geometric stiffness


    FlexibleCorotationalFEMForceField()
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

        m_deformationDofs = core::objectmodel::New< DeformationDofs >();
        m_gaussPointSampler = core::objectmodel::New< GaussPointSampler >();
    }

    virtual ~FlexibleCorotationalFEMForceField() {}


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




}; // class FlexibleCorotationalFEMForceField



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_METAFEMFORCEFIELD_CPP)
extern template class SOFA_Flexible_API FlexibleCorotationalFEMForceField<defaulttype::Vec3Types>;
#endif

} // namespace forcefield
} // namespace component
} // namespace sofa

#endif // FLEXIBLE_METAFEMFORCEFIELD_H
