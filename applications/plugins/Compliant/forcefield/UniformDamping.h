#ifndef SOFA_COMPONENT_DAMPING_UniformDamping_H
#define SOFA_COMPONENT_DAMPING_UniformDamping_H


#include <Compliant/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/helper/template_name.h>

namespace sofa {
namespace component {
namespace forcefield {

/** Damping uniformly applied to all the DOF.
  Each dof represents a constraint violation, and undergoes force \f$ \lambda = -\frac{1}{c} ( x - d v ) \f$, where c is the damping and d the damping ratio.
  */
template<class TDataTypes>
class UniformDamping : public core::behavior::ForceField<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformDamping, TDataTypes),
               SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef core::behavior::ForceField<TDataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    enum { N=DataTypes::deriv_total_size };
    
    Data< Real > damping;    ///< Same compliance applied to all the DOFs
    virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset ) {

        const std::size_t size = this->getMState()->getMatrixSize();

        if(B.compressedMatrix.rows() != int(size)) {
            B.compressedMatrix.resize(size, size);

            struct iterator {
                std::size_t index;
                const Real val;

                iterator& operator++() { ++index; return *this; }

                int col() const { return index; }
                int row() const { return index; }
                SReal value() const { return val; }                                
                const iterator* operator->() const { return this; }
                
                bool operator!=(const iterator& other) const { 
                    return index != other.index;
                }
            };

            iterator first{0, -damping.getValue()}, last{size, {}};
            B.compressedMatrix.setFromTriplets(first, last);
        }

        if(damping.getValue()) {
            B.addToBaseMatrix( matrix, bFact, offset );
        }
    }



    UniformDamping( core::behavior::MechanicalState<DataTypes> *mm = NULL)
        : Inherit( mm )
        , damping( initData(&damping, Real(0), "damping", "uniform viscous damping")) {
        this->isCompliance.setValue(false);
    }
    
        
    
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &,
                          const DataVecDeriv &) { }

    virtual void addDForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &) { }

    virtual void addClambda(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &, SReal) { }

    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*) {
        return nullptr;
    }
    
    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix *,  SReal , unsigned int & ) { }
    virtual SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }


    // template representation
    static std::string templateName(const UniformDamping* self) {
        return helper::template_name(self);
    }

    std::string getTemplateName() const {
        return templateName(this);
    }
    
    
protected:


    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> matrix_type;
    matrix_type B; /// damping matrix (Negative S.D.)
    
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H


