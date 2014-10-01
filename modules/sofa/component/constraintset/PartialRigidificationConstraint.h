#ifndef PARTIALRIGIDIFICATIONCONSTRAINT_H
#define PARTIALRIGIDIFICATIONCONSTRAINT_H

#include <sofa/core/behavior/Constraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/defaulttype/Vec.h>
namespace sofa
{

namespace component
{

namespace constraintset
{

class PartialRigidificationConstraintResolution6Dof : public core::behavior::ConstraintResolution
{
public:

	PartialRigidificationConstraintResolution6Dof() { }
    virtual void init(int /*line*/, double** , double *)
	{
	}

    virtual void initForce(int , double* )
	{
	}

    virtual void resolution(int , double** /*w*/, double* , double* )
	{
	}

    void store(int , double* , bool /*convergence*/)
	{
	}


};

template< class DataTypes >
class PartialRigidificationConstraint : public core::behavior::Constraint<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(PartialRigidificationConstraint,DataTypes), SOFA_TEMPLATE(core::behavior::Constraint,DataTypes));

	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecDeriv VecDeriv;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef typename DataTypes::MatrixDeriv MatrixDeriv;
	typedef typename Coord::value_type Real;
    typedef sofa::defaulttype::Vec<3, Real> Vec3;
	typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
	typedef typename core::behavior::Constraint<DataTypes> Inherit;

	typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
	typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
	typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
	typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:

	unsigned int cid;



	PartialRigidificationConstraint(MechanicalState* object)
		: Inherit(object)
	{
	}


	PartialRigidificationConstraint()
	{
	}

	virtual ~PartialRigidificationConstraint() {}
public:
	virtual void init();
	virtual void buildConstraintMatrix(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &x);
	virtual void getConstraintViolation(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v);

	virtual void getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset);
};
} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // PARTIALRIGIDIFICATIONCONSTRAINT_H
