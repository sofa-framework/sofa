/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef RIGIDTOAFFINEMULTIMAPPING_H
#define RIGIDTOAFFINEMULTIMAPPING_H

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/BaseMatrix.h> 
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <SofaBaseMechanics/MechanicalObject.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/Multi2Mapping.inl>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>

#include <RigidScale/mapping/RigidScaleMappingJacobian.h>

namespace sofa
{
namespace component
{
namespace mapping
{
using namespace sofa::defaulttype;


/**
 * @author Ali Dicko @date 2015
 */
template <class In1, class In2, class Out>
class RigidScaleToAffineMultiMapping : public core::Multi2Mapping<In1, In2, Out>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE3(RigidScaleToAffineMultiMapping, In1, In2, Out), SOFA_TEMPLATE3(core::Multi2Mapping, In1, In2, Out));

    typedef typename core::Multi2Mapping<In1, In2, Out>  Inherit;

    typedef component::container::MechanicalObject<Out> OutType;
    typedef component::container::MechanicalObject<In1> InType1;
    typedef component::container::MechanicalObject<In2> InType2;

    typedef typename In1::Real Real;
    typedef typename In1::Coord InCoord1;
	typedef typename In1::Deriv InDeriv1;
	typedef typename In1::VecCoord InVecCoord1;
	typedef typename In1::VecDeriv InVecDeriv1;
    typedef typename In1::MatrixDeriv InMatrixDeriv1;
    typedef typename Inherit::In1DataVecCoord In1DataVecCoord;
    typedef typename Inherit::In1DataVecDeriv In1DataVecDeriv;
    typedef typename Inherit::In1DataMatrixDeriv In1DataMatrixDeriv;

	typedef typename In2::Coord InCoord2;
	typedef typename In2::Deriv InDeriv2;
    typedef typename In2::VecCoord InVecCoord2;
    typedef typename In2::VecDeriv InVecDeriv2;
    typedef typename In2::MatrixDeriv InMatrixDeriv2;
    typedef typename Inherit::In2DataVecCoord In2DataVecCoord;
    typedef typename Inherit::In2DataVecDeriv In2DataVecDeriv;
    typedef typename Inherit::In2DataMatrixDeriv In2DataMatrixDeriv;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Inherit::OutDataVecCoord OutDataVecCoord;
    typedef typename Inherit::OutDataVecDeriv OutDataVecDeriv;
    typedef typename Inherit::OutDataMatrixDeriv OutDataMatrixDeriv;

    typedef Mat<OutDeriv::total_size,InDeriv1::total_size, SReal> MatBlock1;
	typedef Mat<OutDeriv::total_size, InDeriv2::total_size, SReal> MatBlock2;

    typedef component::linearsolver::EigenSparseMatrix<In1, Out> SparseJMatrixEigen1;
    typedef component::linearsolver::EigenSparseMatrix<In2, Out> SparseJMatrixEigen2;
    typedef linearsolver::EigenSparseMatrix<In1,In1> SparseKMatrixEigen1;
    typedef linearsolver::EigenSparseMatrix<In2,In2> SparseKMatrixEigen2;

    typedef helper::vector<defaulttype::BaseMatrix*> jacobianMatrices;
    typedef helper::vector<defaulttype::BaseMatrix*> stiffnessMatrices;

	/****************** CONSTRUCTOR / DESTRUCTOR ***********************/
    RigidScaleToAffineMultiMapping();
    ~RigidScaleToAffineMultiMapping();

	/************************** SOFA METHOD ****************************/
    void init();
    void reinit();
	void reset();

    using Inherit::apply;
    using Inherit::applyJ;
    using Inherit::applyJT;
    using Inherit::computeAccFromMapping;

    void apply(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */
			   , const helper::vector<OutDataVecCoord*>& /*dataVecOutPos*/
			   , const helper::vector<const In1DataVecCoord*>& /*dataVecIn1Pos*/
			   , const helper::vector<const In2DataVecCoord*>& /*dataVecIn2Pos*/);

    void applyJ(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */
			    , const helper::vector< OutDataVecDeriv*>& /*dataVecOutVel*/		
				, const helper::vector<const In1DataVecDeriv*>& /*dataVecIn1Vel*/ 
				, const helper::vector<const In2DataVecDeriv*>& /*dataVecIn2Vel*/);

    void applyJT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */
				 , const helper::vector< In1DataVecDeriv*>& /*dataVecOut1Force*/
				 , const helper::vector< In2DataVecDeriv*>& /*dataVecOut2Force*/
				 , const helper::vector<const OutDataVecDeriv*>& /*dataVecInForce*/);

    void applyJT(const helper::vector< InMatrixDeriv1*>& /*outConstraint1*/ 
				 , const helper::vector< InMatrixDeriv2*>& /*outConstraint2*/
				 , const helper::vector<const OutMatrixDeriv*>& /*inConstraint*/);
	
	void applyJT(const core::ConstraintParams* /* cparams */
				 , const helper::vector< In1DataMatrixDeriv*>& /* dataMatOut1Const */
				 , const helper::vector< In2DataMatrixDeriv*>&  /* dataMatOut2Const */
				 , const helper::vector<const OutDataMatrixDeriv*>& /* dataMatInConst */);
	
    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/);

    void computeAccFromMapping(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */
							   , const helper::vector< OutDataVecDeriv*>& /*dataVecOutAcc*/
							   , const helper::vector<const In1DataVecDeriv*>& /*dataVecIn1Vel*/
							   , const helper::vector<const In2DataVecDeriv*>& /*dataVecIn2Vel*/
							   , const helper::vector<const In1DataVecDeriv*>& /*dataVecIn1Acc*/
							   , const helper::vector<const In2DataVecDeriv*>& /*dataVecIn2Acc*/);

    void computeAccFromMapping(const core::MechanicalParams* /*mparams*/, OutVecDeriv& /*f*/,const OutVecCoord& /*x*/, const OutVecDeriv& /*v*/);

    const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();

    const sofa::defaulttype::BaseMatrix* getJ();

    const sofa::defaulttype::BaseMatrix* getK();		                   

    Data< helper::vector<unsigned> > index;  ///< Two indices per child: the index of the rigid, and the index of scale
	Data< bool > automaticInit; ///< Automatic setup based only on the rigid position. The index data is init, the scale mechanical state is reinit and the affine mechanical state is reinit.
    Data< bool > useGeometricStiffness; ///< To indication if we use the geometric stiffness

protected:
    /****************************  METHODS ****************************/
	void autoInit();
    void setup();

    void updateK1(SparseKMatrixEigen1& /*stiffness*/, const InVecCoord1& /*vIn1*/, const OutVecDeriv& /*childForce*/);
    void updateK2(SparseKMatrixEigen2& /*stiffness*/, const InVecCoord1& /*vIn1*/, const OutVecDeriv& /*childForce*/);

    void updateJ1(SparseJMatrixEigen1& /*jacobian*/, const InVecCoord1& /*vIn1*/, const InVecCoord2& /*vIn2*/, const OutVecCoord& /*vOut*/);    
	void updateJ2(SparseJMatrixEigen2& /*jacobian*/, const InVecCoord1& /*vIn1*/, const InVecCoord2& /*vIn2*/, const OutVecCoord& /*vOut*/);

	/// computeAffineFromRigidAndScale : function f(r, s) = a. 
	/// r = rigid body defined by (p, q) => p is the position and q is the quarternion which describe the rigid orientation
	/// s = scale matrix, but defined by only the diagonal component (sx, sy, sz)
	/// J1 = df/dr and J2 = df/ds
	void computeAffineFromRigidAndScale(const InCoord1&, const InCoord2&, OutCoord&);

    /*********************** CLASS ATTRIBUTES **************************/
//    SparseKMatrixEigen1 _K1;
//    SparseKMatrixEigen2 _K2;

    SparseJMatrixEigen1 _J1;
    SparseJMatrixEigen2 _J2;
    jacobianMatrices _Js;

    helper::vector<SparseJMatrixEigen1*> _DJ1;

    // In/Out mechanical object
    InType1* stateIn1;
    InType2* stateIn2;
    OutType* stateOut;
};

}//namespace mapping
}// namespace component
}//namespace sofa


#endif
