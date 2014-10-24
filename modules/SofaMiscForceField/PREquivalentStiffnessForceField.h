#ifndef PREQUIVALENTSTIFFNESSFORCEFIELD_H
#define PREQUIVALENTSTIFFNESSFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/vector.h>

#include <string>

namespace sofa
{

namespace component
{

namespace forcefield
{


using sofa::helper::vector;
using sofa::core::MechanicalParams;
using sofa::defaulttype::BaseMatrix;

template<typename DataTypes>
class PREquivalentStiffnessForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public :
	SOFA_CLASS(SOFA_TEMPLATE(PREquivalentStiffnessForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef sofa::core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::Real Real;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::Pos Pos;
    typedef typename Coord::Quat Quaternion;
    typedef helper::vector<Coord> VecCoord;
    typedef helper::vector<Deriv> VecDeriv;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;

    typedef sofa::defaulttype::Vec<3, Real> Vec3;
    typedef sofa::defaulttype::Vec<6, Real> Vec6;
    typedef sofa::defaulttype::Vec<12, Real> Vec12;
    typedef sofa::defaulttype::Mat<3, 3, Real> Mat33;
    typedef sofa::defaulttype::Mat<3, 3, Real> Mat44;
    typedef sofa::defaulttype::Mat<6, 6, Real> Mat66;
    typedef sofa::defaulttype::Mat<12, 12, Real> Mat12x12;

    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<Mat66> CSRMatB66;

    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat66>::ColBlockConstIterator CSRMatB66ColBlockIter;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat66>::RowBlockConstIterator CSRMatB66RowBlockIter;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat66>::BlockConstAccessor CSRMatB66BlockConstAccessor;

    typedef helper::vector<Mat66> VecMat66;
    typedef helper::vector<Mat12x12> VecMat12;

public :
	PREquivalentStiffnessForceField();
	virtual ~PREquivalentStiffnessForceField();

    virtual void init();

	virtual void addForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& f , const DataVecCoord& x , const DataVecDeriv& v);
	virtual void addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&   df , const DataVecDeriv&   dx );
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset);

    void displaceFrames(const VecCoord& frames, VecCoord& displaced, const VecDeriv& dq, const Real epsilon);
    void computeForce(const VecCoord& pos, const VecCoord& restPos, VecDeriv& f);

protected :
    Data<std::string> m_complianceFile;
    VecMat66 m_complianceMat;
    VecMat66 m_CInv;
    VecMat66 m_H;
    VecMat12 m_K;
    VecCoord m_pos;
    VecCoord m_restPos;

};


} // forcefield

} // component

} // sofa

#endif // PREQUIVALENTSTIFFNESSFORCEFIELD_H
