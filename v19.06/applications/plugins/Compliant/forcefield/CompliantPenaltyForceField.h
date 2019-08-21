#ifndef SOFA_COMPONENT_COMPLIANCE_COMPLIANTPenaltyFORCEFIELD_H
#define SOFA_COMPONENT_COMPLIANCE_COMPLIANTPenaltyFORCEFIELD_H

#include <Compliant/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include "../utils/edit.h"

namespace sofa
{
namespace component
{
namespace forcefield
{

/** 
        Uniform stiffness applied only to negative (violated) dofs.

        @author: Matthieu Nesme, 2016
  */

template<class TDataTypes>
class CompliantPenaltyForceField : public core::behavior::ForceField<TDataTypes>
{
public:
    
    SOFA_CLASS(SOFA_TEMPLATE(CompliantPenaltyForceField, TDataTypes),
               SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));


    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;


    Data<SReal> d_stiffness; ///< uniform stiffness value applied to all the DOF
    Data<SReal> d_damping; ///< uniform viscous damping


    virtual SReal getPotentialEnergy( const core::MechanicalParams* /*mparams*/,
                                      const DataVecCoord& x ) const
    {
        const VecCoord& _x = x.getValue();
        const SReal& k = d_stiffness.getValue();

        SReal e = 0;
        for( unsigned int i=0 ; i<_x.size() ; ++i )
        {
            if( _x[i][0] > 0 )
            {
                e += .5 * k * _x[i][0]*_x[i][0];
            }
        }
        return e;
    }
    
    virtual void addForce(const core::MechanicalParams *,
                          DataVecDeriv &f,
                          const DataVecCoord &x,
                          const DataVecDeriv &)
    {
        const VecCoord& _x = x.getValue();
        VecDeriv& _f = *f.beginEdit();

        const SReal& k = d_stiffness.getValue();
        m_violated.resize(_x.size());

        for( unsigned int i=0 ; i<_x.size() ; ++i )
        {
            if( _x[i][0] < 0 )
            {
                _f[i][0] -= k * _x[i][0];
                m_violated[i] = true;
            }
            else
                m_violated[i] = false;
        }

        f.endEdit();
    }

    virtual void addDForce(const core::MechanicalParams *mparams,
                           DataVecDeriv &df,
                           const DataVecDeriv &dx)
    {

        const VecDeriv& _dx = dx.getValue();
        VecDeriv& _df = *df.beginEdit();

        const SReal k = d_stiffness.getValue() * (SReal)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
        const SReal b = d_damping.getValue() * (SReal)mparams->bFactor();


        for( unsigned int i=0 ; i<_dx.size() ; ++i )
        {
            if( m_violated[i] )
            {
                _df[i][0] -= k * _dx[i][0];
                _df[i][0] -= b * _dx[i][0];
            }
        }

        df.endEdit();
    }

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix,
                               SReal kFact, unsigned int &offset )
    {
        const SReal& k = - d_stiffness.getValue() * kFact;
        size_t size = this->mstate->getMatrixSize();

        for( unsigned int i=0 ; i<size ; ++i )
        {
            if( m_violated[i] )
            {
                matrix->add( offset+i, offset+i, k );
            }
        }
    }

    virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix,
                               SReal bFact, unsigned int &offset )
    {
        const SReal& b = - d_damping.getValue() * bFact;
        size_t size = this->mstate->getMatrixSize();

        for( unsigned int i=0 ; i<size ; ++i )
        {
            if( m_violated[i] )
            {
                matrix->add( offset+i, offset+i, b );
            }
        }
    }

    virtual void updateForceMask()
    {
        for( unsigned int i=0 ; i<m_violated.size() ; ++i )
            if(m_violated[i])
                this->mstate->forceMask.insertEntry(i);
    }


protected:

    CompliantPenaltyForceField( core::behavior::MechanicalState<DataTypes> *mm = 0)
        : Inherit1(mm)
        , d_stiffness( initData(&d_stiffness, "stiffness", "uniform stiffness value applied to all the DOF"))
        , d_damping( initData(&d_damping, SReal(0), "damping", "uniform viscous damping"))
    {
        this->isCompliance.setValue(false);
    }

    helper::vector<bool> m_violated;

};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_COMPLIANTPenaltyFORCEFIELD_H


