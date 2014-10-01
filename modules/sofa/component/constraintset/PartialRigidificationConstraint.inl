#ifndef PARTIALRIGIDIFICATIONCONSTRAINT_INL
#define PARTIALRIGIDIFICATIONCONSTRAINT_INL

#include <sofa/component/constraintset/PartialRigidificationConstraint.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using defaulttype::Rigid3dTypes;

template<class DataTypes>
void PartialRigidificationConstraint<DataTypes>::init()
{
    this->mstate = dynamic_cast<MechanicalState*>(this->getContext()->getMechanicalState());
    assert(this->mstate);


}


template<class DataTypes>
void PartialRigidificationConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/ /* PARAMS FIRST */, defaulttype::BaseVector */*resV*/, const DataVecCoord & /*x*/ , const DataVecDeriv &/*v*/)
{

}


template<class DataTypes>
void PartialRigidificationConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/ /* PARAMS FIRST */, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &x)
{

    std::cout<<"step0"<<std::endl;
	const VecCoord X = x.getValue();
	MatrixDeriv& constraints = *c_d.beginEdit();
	cid = cIndex;
    std::cout<<"step1"<<std::endl;
    const Vec3 cx(1, 0, 0), cy(0, 1, 0), cz(0, 0, 1), vZero(0, 0, 0);
  //  const Vec3 p0p1 = X[1].getCenter() - X[0].getCenter();

//	const Deriv::Rot tau0p1cx = p0p1.cross(-cx);
//	const Deriv::Rot tau0p1cy = p0p1.cross(-cy);
//	const Deriv::Rot tau0p1cz = p0p1.cross(-cz);

    std::cout<<"step2"<<std::endl;

	MatrixDerivRowIterator cit = constraints.writeLine(cIndex);
    cit.addCol(0, Deriv(-cx, vZero));
//	cit.addCol(0, Deriv(cx, tau0p1cx));

	cit = constraints.writeLine(cIndex+1);
	cit.setCol(0, Deriv(-cy, vZero));
//	cit.addCol(0, Deriv(cy, tau0p1cy));

	cit = constraints.writeLine(cIndex+2);
	cit.setCol(0, Deriv(-cz, vZero));
//	cit.addCol(0, Deriv(cz, tau0p1cz));

	cit = constraints.writeLine(cIndex+3);
	cit.setCol(0, Deriv(vZero, -cx));
//    cit.addCol(0, Deriv(cx.cross(p0p1), vZero));

	cit = constraints.writeLine(cIndex+4);
	cit.setCol(0, Deriv(vZero, -cy));
//    cit.addCol(0, Deriv(cy.cross(p0p1), vZero));

	cit = constraints.writeLine(cIndex+5);
	cit.setCol(0, Deriv(vZero, -cz));
//    cit.addCol(0, Deriv(cz.cross(p0p1), vZero));

      std::cout<<"DEBUG/// state"<<this->mstate->getName()<<std::endl;

      // Indicates the size of the constraint block
      cIndex +=6;
}

template<class DataTypes>
void PartialRigidificationConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
//	for(size_t i = 0 ; i < 6 ; ++i)
    resTab[offset] = new PartialRigidificationConstraintResolution6Dof();

    // indicates the size of the block on which the constraint resoluation works
    offset += 6;
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // PARTIALRIGIDIFICATIONCONSTRAINT_INL
