#ifndef FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H
#define FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H


#include "../shapeFunction/BarycentricShapeFunction.h"
#include "../quadrature/TopologyGaussPointSampler.h"
#include "../deformationMapping/LinearMapping.h"
#include "../strainMapping/CorotationalStrainMapping.h"
#include "../material/HookeForceField.h"

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class SOFA_Flexible_API FlexibleTetrahedronFEMForceField : virtual public core::behavior::ForceField<DataTypes>, virtual public shapefunction::BarycentricShapeFunction<core::behavior::ShapeFunction3>
{
public:

    typedef core::behavior::ForceField<DataTypes> Inherit;

    SOFA_CLASS2(SOFA_TEMPLATE(FlexibleTetrahedronFEMForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField,DataTypes),SOFA_TEMPLATE(shapefunction::BarycentricShapeFunction,core::behavior::ShapeFunction3));

    virtual std::string getTemplateName() const { return Inherit::getTemplateName(); }
    static std::string templateName( const FlexibleTetrahedronFEMForceField<DataTypes>* = NULL) { return DataTypes::Name(); }

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


        core::topology::BaseMeshTopology *topo = NULL;
        this->getContext()->get( topo, core::objectmodel::BaseContext::SearchUp );
        if( !topo ) { serr<<"No MeshTopology found"<<sendl; return; }


/// ShapeFunction
        ShapeFunction::_state = this->mstate;
        ShapeFunction::parentTopology = topo;
        ShapeFunction::init();


/// GaussPointSampler
        GaussPointSampler* gaussPointSampler  = static_cast< GaussPointSampler* >( _baseGaussPointSampler.get() );
        gaussPointSampler->f_method.setValue( 0 );
        gaussPointSampler->f_order.setValue( 1 );
        gaussPointSampler->parentTopology = topo;
        gaussPointSampler->f_inPosition.setValue( static_cast<topology::MeshTopology*>(topo)->seqPoints.getValue() );
        gaussPointSampler->init();


/// DeformationMapping
        DeformationDofs* deformationDofs = static_cast< DeformationDofs* >( _baseDeformationDofs.get() );
        _baseDeformationMapping = core::objectmodel::New< DeformationMapping >( this->mstate, deformationDofs );
        DeformationMapping* deformationMapping = static_cast< DeformationMapping* >( _baseDeformationMapping.get() );
        deformationMapping->_sampler = gaussPointSampler;
        deformationMapping->_shapeFunction = this;
        deformationMapping->init( false );

        unsigned size = gaussPointSampler->getNbSamples();



/// Strain Mapping
        _strainJacobianBlocks.resize( size );


/// Material
        _materialBlocks.resize( size );
        for( unsigned int i=0 ; i<size ; i++ ) _materialBlocks[i].volume=&gaussPointSampler->f_volume.getValue()[i];



        Inherit::init();

        reinit();
    }

    virtual void reinit()
    {
        _lambda = _youngModulus.getValue()*_poissonRatio.getValue()/((1-2*_poissonRatio.getValue())*(1+_poissonRatio.getValue()));
        _mu2    = _youngModulus.getValue()/(1+_poissonRatio.getValue());

        for( unsigned i=0; i < _materialBlocks.size() ; ++i )
        {
            //_strainJacobianBlocks[i].init();
            _materialBlocks[i].init( _youngModulus.getValue(), _poissonRatio.getValue(), _lambda, _mu2, _viscosity.getValue() );
        }

        // reinit matrices
        //if(this->assembleC.getValue()) updateC();
        //if(this->assembleK.getValue()) updateK();
        //if(this->assembleB.getValue()) updateB();

        Inherit::reinit();
    }

    virtual void addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v)
    {
        if( isCompliance.getValue() ) return; // if seen as a compliance, then apply no force directly, they will be applied as constraints in writeConstraints

        VecDeriv&  f = *_f.beginEdit();
        const VecCoord&  x = _x.getValue();
        const VecDeriv&  v = _v.getValue();

        DeformationMapping* deformationMapping = static_cast< DeformationMapping* >( _baseDeformationMapping.get() );
        typename DeformationMapping::SparseMatrix& deformationJacobianBlocks = deformationMapping->getJacobianBlocks();

        // TODO use masks
        for( unsigned i=0; i < _strainJacobianBlocks.size() ; ++i )
        {
            defaulttype::F331Types::Coord F;
            defaulttype::F331Types::Deriv VF;
            defaulttype::F331Types::Deriv PF;

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addapply( F, x[index] );
                deformationJacobianBlocks[i][j].addmult( VF, v[index] );
            }

            defaulttype::E331Types::Coord E;
            defaulttype::E331Types::Deriv VE;
            defaulttype::E331Types::Deriv PE;

            _strainJacobianBlocks[i].addapply_svd( E, F );
            _strainJacobianBlocks[i].addmult( VE, VF );

            _materialBlocks[i].addForce( PE, E, VE );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addMultTranspose( f[index], PF );
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

    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&   _df , const DataVecDeriv&   _dx )
    {
        if( isCompliance.getValue() ) return; // if seen as a compliance, then apply no force directly, they will be applied as constraints


        VecDeriv&  df = *_df.beginEdit();
        const VecDeriv&  dx = _dx.getValue();


        DeformationMapping* deformationMapping = static_cast< DeformationMapping* >( _baseDeformationMapping.get() );
        typename DeformationMapping::SparseMatrix& deformationJacobianBlocks = deformationMapping->getJacobianBlocks();

        // TODO use masks
        for( unsigned i=0; i < _strainJacobianBlocks.size() ; ++i )
        {
            defaulttype::F331Types::Coord F;
            defaulttype::F331Types::Deriv PF;

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = deformationMapping->f_index.getValue()[i][j];
                deformationJacobianBlocks[i][j].addmult( F, dx[index] );
            }

            defaulttype::E331Types::Coord E;
            defaulttype::E331Types::Deriv PE;

            _strainJacobianBlocks[i].addmult( E, F );

            _materialBlocks[i].addDForce( PE, E, mparams->kFactor(), mparams->bFactor() );

            _strainJacobianBlocks[i].addMultTranspose( PF, PE );

            for( unsigned int j=0 ; j<deformationJacobianBlocks[i].size() ; j++ )
            {
                unsigned int index = deformationMapping->f_index.getValue()[i][j];
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

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }
    //@}



    typedef shapefunction::BarycentricShapeFunction<core::behavior::ShapeFunction3> ShapeFunction;
    typedef engine::TopologyGaussPointSampler GaussPointSampler;
    typedef mapping::LinearMapping< DataTypes, defaulttype::F331Types > DeformationMapping;
    //typedef mapping::CorotationalStrainMapping< defaulttype::F331Types, defaulttype::E331Types > StrainMapping;
    //typedef HookeForceField< defaulttype::E331Types > MaterialForceField;

    typedef component::container::MechanicalObject < defaulttype::F331Types > DeformationDofs;
    //typedef component::container::MechanicalObject < defaulttype::E331Types > StrainDofs;

    //typedef defaulttype::LinearJacobianBlock< defaulttype::F331Types, defaulttype::E331Types > DeformationJacobianBlock;
    //typedef vector< DeformationJacobianBlock >  DeformationJacobianBlocks;
    //DeformationJacobianBlocks _deformationJacobianBlocks;

    typedef defaulttype::CorotationalStrainJacobianBlock< defaulttype::F331Types, defaulttype::E331Types > StrainJacobianBlock;
    typedef vector< StrainJacobianBlock >  StrainJacobianBlocks;
    StrainJacobianBlocks _strainJacobianBlocks;

	typedef defaulttype::IsotropicHookeLaw<typename defaulttype::E331Types::Real, defaulttype::E331Types::material_dimensions, defaulttype::E331Types::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock< defaulttype::E331Types, LawType > MaterialBlock;
    typedef vector< MaterialBlock >  MaterialBlocks;
    MaterialBlocks _materialBlocks;



    /** @name  Material parameters */
    //@{
    Data<Real> _youngModulus;
    Data<Real> _poissonRatio;
    Data<Real> _viscosity;
    Real _lambda;  ///< Lamé first coef
    Real _mu2;     ///< Lamé second coef * 2
    //@}




protected:


    core::BaseMapping::SPtr _baseDeformationMapping;
    BaseMechanicalState::SPtr _baseDeformationDofs;
    //shapefunction::BaseShapeFunction<core::behavior::ShapeFunction3>::SPtr _baseShapeFunction;
    engine::BaseGaussPointSampler::SPtr _baseGaussPointSampler;

    Data< bool > isCompliance;  ///< Consider as compliance, else consider as stiffness

    FlexibleTetrahedronFEMForceField( MStateType *mm = NULL )
        : Inherit( mm )
        //, assembleC ( initData ( &assembleC,false, "assembleC","Assemble the Compliance matrix" ) )
        //, assembleK ( initData ( &assembleK,false, "assembleK","Assemble the Stiffness matrix" ) )
        //, assembleB ( initData ( &assembleB,false, "assembleB","Assemble the Damping matrix" ) )
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,(Real)0.45f,"poissonRatio","Poisson Ratio"))
        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
        , isCompliance( initData(&isCompliance, false, "isCompliance", "Consider the component as a compliance, else as a stiffness"))
    {
        _baseDeformationDofs = core::objectmodel::New< DeformationDofs >();
        _baseGaussPointSampler = core::objectmodel::New< GaussPointSampler >();
        //_baseShapeFunction = core::objectmodel::New< ShapeFunction >();

    }

    virtual ~FlexibleTetrahedronFEMForceField() {}


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




}; // class FlexibleTetrahedronFEMForceField



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_CPP)
#ifdef SOFA_FLOAT
extern template class SOFA_Flexible_API FlexibleTetrahedronFEMForceField<defaulttype::Vec3fTypes>;
#else
extern template class SOFA_Flexible_API FlexibleTetrahedronFEMForceField<defaulttype::Vec3dTypes>;
#endif
#endif

} // namespace forcefield
} // namespace component
} // namespace sofa

#endif // FLEXIBLE_TETRAHEDRONFEMFORCEFIELD_H
