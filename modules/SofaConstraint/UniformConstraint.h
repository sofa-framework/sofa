#ifndef ISPHYSICS_INTERACTION_UNIFORMCONSTRAINT_H
#define ISPHYSICS_INTERACTION_UNIFORMCONSTRAINT_H

#include <sofa/core/behavior/Constraint.h>



namespace isphysics
{
namespace interaction
{

template < class DataTypes >
class UniformConstraint : public sofa::core::behavior::Constraint< DataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformConstraint, DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Constraint, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real  Real;

    typedef sofa::Data<VecCoord>    DataVecCoord;
    typedef sofa::Data<VecDeriv>    DataVecDeriv;
    typedef sofa::Data<MatrixDeriv> DataMatrixDeriv;

    void buildConstraintMatrix(const sofa::core::ConstraintParams* cParams, DataMatrixDeriv & c, unsigned int &cIndex, const DataVecCoord &x) override;

    void getConstraintViolation(const sofa::core::ConstraintParams* cParams, sofa::defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) override;

    void getConstraintResolution(const sofa::core::ConstraintParams* cParams, std::vector<sofa::core::behavior::ConstraintResolution*>& crVector, unsigned int& offset) override;

    sofa::Data<sofa::helper::vector<Real> > d_softW;
    sofa::Data<bool> d_iterative;
protected:

    unsigned m_constraintIndex;

    UniformConstraint();
};


}

}

#endif // ISPHYSICS_INTERACTION_UNIFORMCONSTRAINT_H
