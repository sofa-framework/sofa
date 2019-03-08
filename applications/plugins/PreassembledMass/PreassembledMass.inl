#ifndef __PreassembledMass_INL
#define __PreassembledMass_INL

#include "PreassembledMass.h"

#include <sofa/core/objectmodel/BaseNode.h>

#include <Compliant/assembly/AssemblyVisitor.h>

namespace sofa
{

namespace component
{

namespace mass
{

/// removing no longer necessary mapped masses
class RemoveChildMassVisitor : public simulation::Visitor
{
public:
    RemoveChildMassVisitor(const core::ExecParams* params)
        : simulation::Visitor(params)
    {
    }

    Result processNodeTopDown(simulation::Node* node) override
    {
        core::behavior::BaseMass* mass;
        node->get(mass);
        if( !dynamic_cast<BasePreassembledMass*>(mass) )
            node->removeObject(mass);
        return RESULT_CONTINUE;
    }

};



unsigned int BasePreassembledMass::s_instanciationCounter = 0;


template < class DataTypes >
void PreassembledMass< DataTypes >::init()
{
    core::behavior::Mass<DataTypes>::init();
}

template < class DataTypes >
void PreassembledMass< DataTypes >::bwdInit()
{

    MassMatrix& massMatrix = *d_massMatrix.beginEdit();

    // if the mass matrix is not given manually
    if( massMatrix.rows() != massMatrix.cols() || massMatrix.rows()!=(typename MassMatrix::Index)this->mstate->getMatrixSize() )
    {
        // perform assembly
        core::MechanicalParams mparams = *core::MechanicalParams::defaultInstance();
        mparams.setKFactor(0);
        mparams.setBFactor(0);
        mparams.setMFactor(1);
        mparams.setDt( this->getContext()->getDt() ); // should not be used but to be sure

        simulation::AssemblyVisitor assemblyVisitor( &mparams );
        this->getContext()->executeVisitor( &assemblyVisitor );
        component::linearsolver::AssembledSystem sys;
        assemblyVisitor.assemble( sys );
        massMatrix.compressedMatrix = sys.H;

        if( massMatrix.rows()!=(typename MassMatrix::Index)this->mstate->getMatrixSize() )
        {
            serr<<"Are you sure that every independent dofs are in independent graph branches?\n";
            assert(false);
        }

        if( _instanciationNumber == 0 ) // only the first one (last bwdInit called) will call the mass removal
        {
    //        std::cerr<<SOFA_CLASS_METHOD<<"removing child masses"<<std::endl;

            // visitor to delete child mass
            RemoveChildMassVisitor removeChildMassVisitor( core::ExecParams::defaultInstance() );
            this->getContext()->executeVisitor( &removeChildMassVisitor );

            typename LinkMassNodes::Container massNodes = l_massNodes.getValue();
            for ( unsigned int i = 0; i < massNodes.size() ; i++)
            {
               if( massNodes[i]->isActive() ) massNodes[i]->setActive( false );
            }
        }
    }


    // for human debug
    Real totalmass = 0;
    for(typename MassMatrix::Index r=0;r<massMatrix.rows();++r)
        for(typename MassMatrix::Index c=0;c<massMatrix.cols();++c)
            totalmass+=massMatrix.element(r,c);
    sout<<"total mass: "<<totalmass/this->mstate->getMatrixBlockSize()<<sendl;

    d_massMatrix.endEdit();
}





////////////////////////////////////





// -- Mass interface
template < class DataTypes >
void PreassembledMass< DataTypes >::addMDx( const core::MechanicalParams*, DataVecDeriv& res, const DataVecDeriv& dx, double factor )
{
    if( factor == 1.0 )
    {
        d_massMatrix.getValue().addMult( res, dx );
    }
    else
    {
        d_massMatrix.getValue().addMult( res, dx, factor );
    }
}

template < class DataTypes >
void PreassembledMass< DataTypes >::accFromF( const core::MechanicalParams*, DataVecDeriv& /*acc*/, const DataVecDeriv& /*f*/ )
{
    serr<<"accFromF not yet implemented (the matrix inversion is needed)"<<sendl;
}

template < class DataTypes >
double PreassembledMass< DataTypes >::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& v ) const
{
    const VecDeriv& _v = v.getValue();
    double e = 0;

    VecDeriv Mv;
    d_massMatrix.getValue().mult( Mv, _v );

    for( unsigned int i=0 ; i<_v.size() ; i++ )
        e += _v[i] * Mv[i];

    return e/2;
}

template < class DataTypes >
double PreassembledMass< DataTypes >::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x ) const
{
    serr<<SOFA_CLASS_METHOD<<"not implemented!\n";
    return core::behavior::Mass< DataTypes >::getPotentialEnergy( mparams, x );

//    const VecCoord& _x = x.getValue();

//    VecCoord Mx/* = d_massMatrix * _x*/;
//    d_massMatrix.mult( Mx, _x );

//    SReal e = 0;
//    // gravity
//    Vec3d g ( this->getContext()->getGravity() );
//    Deriv theGravity;
//    DataTypes::set ( theGravity, g[0], g[1], g[2] );
//    for( unsigned int i=0 ; i<_x.size() ; i++ )
//    {
//        e -= theGravity*Mx[i];
//    }
//    return e;
}



template < class DataTypes >
void PreassembledMass< DataTypes >::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        Vec3 g ( this->getContext()->getGravity() * (mparams->dt()) );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (mparams->dt());

        // add weight force
        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
        d_v.endEdit();
    }
}


template < class DataTypes >
void PreassembledMass< DataTypes >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue()) return;

    VecDeriv& _f = *f.beginEdit();

    // gravity
    Vec3 g ( this->getContext()->getGravity() );
//    Deriv theGravity;
//    DataTypes::set( theGravity, g[0], g[1], g[2] );
    // add weight
//    d_massMatrix.template addMul_by_line<Real,VecDeriv,Deriv>( _f, theGravity );

    //TODO optimize this!!!
    VecDeriv gravities(_f.size());
    for(size_t i=0 ; i<_f.size() ; ++i )
        DataTypes::set( gravities[i], g[0], g[1], g[2] );
    d_massMatrix.getValue().addMult( _f, gravities );

    f.endEdit();
}

template < class DataTypes >
void PreassembledMass< DataTypes >::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
    d_massMatrix.getValue().addToBaseMatrix( r.matrix, mFactor, r.offset );
}

} // namespace mass

} // namespace component

} // namespace sofa

#endif // __PreassembledMass_INL
