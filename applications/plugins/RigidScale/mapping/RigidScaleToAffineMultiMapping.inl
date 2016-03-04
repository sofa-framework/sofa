#ifndef RigidScaleToAffineMultiMapping_INL
#define RigidScaleToAffineMultiMapping_INL

#include <sofa/helper/rmath.h>
#include <Compliant/utils/se3.h>

#include <RigidScale/mapping/RigidScaleToAffineMultiMapping.h>

namespace sofa
{
namespace component
{
namespace mapping
{
    using namespace defaulttype;

//*********************************************************************
// ********************* Constructor / Destructor *********************
//*********************************************************************
template <class I1, class I2, class O>
RigidScaleToAffineMultiMapping<I1,I2,O>::RigidScaleToAffineMultiMapping():
Inherit()
, index(initData(&index, vector<unsigned>(), "index", "list of couples (index in rigid DOF + index in scale with the type Vec3d)"))
, automaticInit(initData(&automaticInit, false, "autoInit", "Init the scale and affine mechanical state, and the index data."))
, useGeometricStiffness(initData(&useGeometricStiffness, false, "useGeometricStiffness", "To specify if the geometric stiffness is used or not."))
, _Js(2)
{
}

template <class I1, class I2, class O>
RigidScaleToAffineMultiMapping<I1,I2,O>::~RigidScaleToAffineMultiMapping()
{
	for (unsigned int i=0; i<_DJ1.size(); i++) delete _DJ1[i];
}

//*********************************************************************
//*************************** SOFA METHODS ****************************
//*********************************************************************
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::init()
{
	// Automatic init if required
	if (this->automaticInit.getValue()) this->autoInit();
	// Init of the different parameter
	this->setup();
	// Call of the parent method
	Inherit::init();
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1, I2, O>::reset()
{
	// Automatic init if required
	if (this->automaticInit.getValue()) this->autoInit();
	// Reset of the different parameter
	this->setup();
	// Call of the parent method
	Inherit::reset();
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::reinit()
{	
	// Update of the different parameter
	this->setup();	
	// Call of the parent method
	Inherit::reinit();
}

//*******************************************************************
// Apply consists in computing A = F([t, R(q)], S) = [t, R(q)*S]
// - t = position of the rigid body
// - R(q) = rotation matrix computed from the rigid quaternion
// - S = scale matrix
//*******************************************************************
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::apply(const core::MechanicalParams* /*mparams*/
											   , const helper::vector<OutDataVecCoord*>& dataVecOutPos
											   , const helper::vector<const In1DataVecCoord*>& dataVecIn1Pos
											   , const helper::vector<const In2DataVecCoord*>& dataVecIn2Pos)
{
	// Index size
	unsigned int ind0, ind1, ind2;
	const vector<unsigned>& index_const = this->index.getValue();
	unsigned int indexSize = (unsigned int)index_const.size() / 3;
	
	// Access to input position
	helper::ReadAccessorVector<helper::vector<const In1DataVecCoord*> > in1(dataVecIn1Pos);
	helper::ReadAccessorVector<helper::vector<const In2DataVecCoord*> > in2(dataVecIn2Pos);
	helper::ReadAccessor<In1DataVecCoord> pIn1(*(in1[0]));
	helper::ReadAccessor<In2DataVecCoord> pIn2(*(in2[0]));

	// Update of the output position
	helper::WriteAccessorVector<OutVecCoord> pOut(*dataVecOutPos[0]->beginEdit());
	for (unsigned int i = 0; i < indexSize; ++i)
	{
		ind0 = index_const[3 * i];
		ind1 = index_const[3 * i + 1];
		ind2 = index_const[3 * i + 2];
		this->computeAffineFromRigidAndScale(pIn1[ind1], pIn2[ind2], pOut[ind0]);
	}
	dataVecOutPos[0]->endEdit();

	// Update of the jacobian matrices with the new positions
	updateJ1(_J1, dataVecIn1Pos[0]->getValue(), dataVecIn2Pos[0]->getValue(), dataVecOutPos[0]->getValue());
	updateJ2(_J2, dataVecIn1Pos[0]->getValue(), dataVecIn2Pos[0]->getValue(), dataVecOutPos[0]->getValue());
}

//*******************************************************************
// ApplyJ consists in computing Vs = J*Vm
// - Vs = velocity of slave node
// - Vm = velocity of master node
//*******************************************************************
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::applyJ(const core::MechanicalParams* /*mparams*/
												, const helper::vector<OutDataVecDeriv*>& dataVecOutVel
												, const helper::vector<const In1DataVecDeriv*>& dataVecIn1Vel
												, const helper::vector<const In2DataVecDeriv*>& dataVecIn2Vel)
{	
	_J1.mult(*dataVecOutVel[0], *dataVecIn1Vel[0]);
	_J2.addMult(*dataVecOutVel[0], *dataVecIn2Vel[0]);
}

//*******************************************************************
// ApplyJT consists in computing Fm = J_transpose*Fs
// - Fs = forces of slave node
// - Fm = forces of master node
//*******************************************************************
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::applyJT(const core::MechanicalParams* /*mparams*/
												, const helper::vector< In1DataVecDeriv*>& dataVecOut1Force
												, const helper::vector< In2DataVecDeriv*>& dataVecOut2Force
												, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
	_J1.addMultTranspose(*dataVecOut1Force[0], *dataVecInForce[0]);
	_J2.addMultTranspose(*dataVecOut2Force[0], *dataVecInForce[0]);
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1, I2, O>::applyJT(const core::ConstraintParams* /* cparams */
												   , const helper::vector< In1DataMatrixDeriv*>& /* dataMatOut1Const */
												   , const helper::vector< In2DataMatrixDeriv*>&  /* dataMatOut2Const */
												   , const helper::vector<const OutDataMatrixDeriv*>& /* dataMatInConst */)
{
	return;
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::applyJT(const helper::vector< InMatrixDeriv1*>& /*outConstraint1*/
												 , const helper::vector< InMatrixDeriv2*>& /*outConstraint2*/ 
												 , const helper::vector<const OutMatrixDeriv*>& /*inConstraint*/)
{
    std::cout<<"applyJT(const helper::vector< In1MatrixDeriv*>& /*outConstraint1*/ ,\
            const helper::vector< In2MatrixDeriv*>& /*outConstraint2*/ ,\
            const helper::vector<const OutMatrixDeriv*>& /*inConstraint*/)\
            NOT IMPLEMENTED BY " << core::objectmodel::BaseClass::decodeClassName(typeid(*this)) << std::endl;
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/)
{ }

	
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::computeAccFromMapping(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */
                                                                , const helper::vector< OutDataVecDeriv*>& /*dataVecOutAcc*/
                                                                , const helper::vector<const In1DataVecDeriv*>& /*dataVecIn1Vel*/
                                                                , const helper::vector<const In2DataVecDeriv*>& /*dataVecIn2Vel*/
                                                                , const helper::vector<const In1DataVecDeriv*>& /*dataVecIn1Acc*/
                                                                , const helper::vector<const In2DataVecDeriv*>& /*dataVecIn2Acc*/)
{ }

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::computeAccFromMapping(const core::MechanicalParams* /*mparams*/, OutVecDeriv& /*f*/, const OutVecCoord& /*x*/, const OutVecDeriv& /*v*/)
{
    serr<<"computeAccFromMapping not implemented!"<<sendl;
}

template <class I1, class I2, class O>
const helper::vector<sofa::defaulttype::BaseMatrix*>* RigidScaleToAffineMultiMapping<I1,I2,O>::getJs()
{
    return &_Js;

}

template <class I1, class I2, class O>
const sofa::defaulttype::BaseMatrix* RigidScaleToAffineMultiMapping<I1,I2,O>::getJ()
{
    return &_J1;
}


template <class I1, class I2, class O>
const sofa::defaulttype::BaseMatrix* RigidScaleToAffineMultiMapping<I1,I2,O>::getK()
{
//    return &_K1;
    return NULL;
}

//*********************************************************************
//************************** OTHERS METHODS ***************************
//*********************************************************************	
template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1, I2, O>::autoInit()
{
	// core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get())
	// Init of the mapping
	this->stateIn1 = dynamic_cast<InType1 *>(this->fromModels1.get(0));
	this->stateIn2 = dynamic_cast<InType2 *>(this->fromModels2.get(0));
	this->stateOut = dynamic_cast<OutType *>(this->toModels.get(0));
	// Lets check the different input / output
	if ((this->stateIn1 && this->stateIn2) && this->stateOut)
	{
		// Variables
		InVecCoord2 _x2;
		OutVecCoord _xout;
		vector<unsigned> _index;

		// Get the position of the inputs
		const InVecCoord1& x1_const = this->stateIn1->read(sofa::core::ConstVecCoordId::position())->getValue();
		unsigned int in1Size = (unsigned int)x1_const.size();
		// Resize of the two mechanical states
		this->stateOut->resize(in1Size);
		this->stateIn2->resize(in1Size);
		// Few tests
		if (in1Size == 0)
		{
            std::cout << "RigidScaleToAffineMultiMapping : Setup failed. There is no more rigid DOFs. " << std::endl;
			return;
		}		
		// Fill
		for (unsigned int i = 0; i < (in1Size); ++i)
		{
			// index
			for (unsigned int j = 0; j < 3; ++j) _index.push_back(i);
			// Scale, Affine
			_x2.push_back(InCoord2(1, 1, 1));
			_xout.push_back(OutCoord());
		}
		// Update of the different data
		this->index.setValue(_index);
		(this->stateIn2->x).setValue(_x2);
		(this->stateIn2->x0).setValue(_x2);
		(this->stateOut->x).setValue(_xout);
		(this->stateOut->x0).setValue(_xout);
	}
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1, I2, O>::setup()
{
	// core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get())
	// Init of the mapping
	this->stateIn1 = dynamic_cast<InType1 *>(this->fromModels1.get(0));
	this->stateIn2 = dynamic_cast<InType2 *>(this->fromModels2.get(0));
	this->stateOut = dynamic_cast<OutType *>(this->toModels.get(0));
	// Lets check the different input / output
	if ((this->stateIn1 && this->stateIn2) && this->stateOut)
	{
		// Get the inputs
		const vector<unsigned>& index_const = this->index.getValue();
		const InVecCoord1& x1_const = this->stateIn1->read(sofa::core::ConstVecCoordId::position())->getValue();
		const InVecCoord2& x2_const = this->stateIn2->read(sofa::core::ConstVecCoordId::position())->getValue();
		const OutVecCoord& xout_const = this->stateOut->read(sofa::core::ConstVecCoordId::position())->getValue();
		// Variables
		unsigned int ind0, ind1, ind2;
		unsigned int indexSize = (unsigned int)index_const.size() / 3;
		unsigned int in1Size = (unsigned int)x1_const.size();
		unsigned int in2Size = (unsigned int)x2_const.size();
		unsigned int outSize = (unsigned int)xout_const.size();
		
		// Check if there are some dofs in the rigid mechanical state
		if ( in1Size == 0 )
		{
            std::cout << "RigidScaleToAffineMultiMapping : Setup failed. There is no more rigid DOFs. " << std::endl;
			return;
		}
		// Check of the index pair size
		if (index_const.size() % 3 != 0 || index_const.size() < 3)
		{
            std::cout << "RigidScaleToAffineMultiMapping : Setup failed. Some affine are connected to only a rigid, while they require a rigid and a scale, check the data index. " << std::endl;
			return;
		}
		// Check of the index pair size
		if (outSize <= 0 || indexSize > outSize)
		{
            if(this->f_printLog.getValue()) std::cout << "RigidScaleToAffineMultiMapping : Warning ! No initial position is found on the output mechanical object. " << std::endl;
			this->stateOut->resize(indexSize);
			outSize = indexSize;
		}
		if (indexSize > in1Size || indexSize > in2Size)
		{
            std::cout << "RigidScaleToAffineMultiMapping : Setup failed. There is more index than existing point. " << std::endl;
			return;
		}

		// Temporary positions
		OutCoord tmp;
		OutVecCoord childCoords(xout_const);

		// Initialization of the jacobian matrix size
		_J1.resizeBlocks(outSize, in1Size);
		_J2.resizeBlocks(outSize, in2Size);
		_Js[0] = &_J1; _Js[1] = &_J2;
        // Initialization of the stiffness matrix size
//        _K1.resizeBlocks(in1Size, in1Size);
//        _K2.resizeBlocks(in2Size, in2Size);
//        _Ks[0] = &_K1; _Ks[1] = &_K2;

		// Init of the output position vector
		for (unsigned int i = 0; i < (indexSize); ++i)
		{
			ind0 = index_const[3 * i];
			ind1 = index_const[3 * i + 1];
			ind2 = index_const[3 * i + 2];
			tmp = xout_const[ind0];
			this->computeAffineFromRigidAndScale(x1_const[ind1], x2_const[ind2], tmp);
			childCoords[ind0] = tmp;
		}
		(this->stateOut->x).setValue(childCoords);
		(this->stateOut->x0).setValue(childCoords);
	}
    else std::cout << "RigidScaleToAffineMultiMapping : setup failed" << std::endl;
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1, I2, O>::computeAffineFromRigidAndScale(const InCoord1& in1, const InCoord2& in2, OutCoord& out)
{
	// Data
	OutCoord output;
	Matrix3 rot, scale, affine;

	// Get important components
	Rigid3Types::Quat q = in1.getOrientation();

	// Conversion of the rigid quaternion into a rotation matrix
	q.toMatrix(rot);
	// Conversion of the scale into a 3x3 matrix
	for (unsigned int i = 0; i < 3; ++i) scale[i][i] = in2[i];

	// Computation of the affine matrix without the translation
	affine = rot*scale;

	// --- Translation
	for (unsigned int i = 0; i < 3; ++i) output[i] = in1.getCenter()[i];
	
	// --- Rotation and scale, no shear :-) !
	for (unsigned int i = 0; i < 3; ++i)
		for (unsigned int j = 0; j < 3; ++j)
			output[3 + (3 * i + j)] = affine[i][j];
	out = output;
	return;
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::updateJ1(SparseJMatrixEigen1& _J, const InVecCoord1& vIn1, const InVecCoord2& vIn2, const OutVecCoord& vOut)
{  
	// variable
	MatBlock1 matBlock;
	
	// Cleaning of previous J
	_J.clear();

	// index size
	unsigned int ind0, ind1, ind2;
	const vector<unsigned>& index_const = this->index.getValue();
	unsigned int indexSize = (unsigned int)index_const.size() / 3;
	for (unsigned int i = 0; i < indexSize; ++i)
	{
		// Computation of the new jacobian matrix
		ind0 = index_const[3 * i];
		ind1 = index_const[3 * i + 1];
		ind2 = index_const[3 * i + 2];
		// Begin of block writing
		_J.beginBlockRow(ind0);
		// Clean of the tmp jacobian matrix
		matBlock.clear();
		computeFrameJacobianR(vIn1[ind1], vIn2[ind2], vOut[ind0], matBlock);
        // Creation of the jacobian block(ind0, ind1)
		_J.createBlock(ind1, matBlock);
		// Block writing end
		_J.endBlockRow();		
	}
	_J.compress();
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::updateJ2(SparseJMatrixEigen2& _J, const InVecCoord1& vIn1, const InVecCoord2& vIn2, const OutVecCoord& vOut)
{		
	// variable
	MatBlock2 matBlock;

	// Cleaning of previous J
	_J.clear();

	// index size
	unsigned int ind0, ind1, ind2;
	const vector<unsigned>& index_const = this->index.getValue();
	unsigned int indexSize = (unsigned int)index_const.size() / 3;
	for (unsigned int i = 0; i < indexSize; ++i)
	{
		// Computation of the new jacobian matrix
		ind0 = index_const[3 * i];
		ind1 = index_const[3 * i + 1];
		ind2 = index_const[3 * i + 2];
		// Begin of block writing
		_J.beginBlockRow(ind0);
		// Clean of the tmp jacobian matrix
		matBlock.clear();
		computeFrameJacobianS(vIn1[ind1], vIn2[ind2], vOut[ind0], matBlock);
        // Creation of the jacobian block(ind0, ind2)
		_J.createBlock(ind2, matBlock);
		// Block writing end
		_J.endBlockRow();
	}
	_J.compress();
}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::updateK1(SparseKMatrixEigen1& /*stiffness*/, const InVecCoord1& /*vIn1*/, const OutVecDeriv& /*childForce*/)
{}

template <class I1, class I2, class O>
void RigidScaleToAffineMultiMapping<I1,I2,O>::updateK2(SparseKMatrixEigen2& /*stiffness*/, const InVecCoord1& /*vIn1*/, const OutVecDeriv& /*childForce*/)
{}

}// namespace mapping
}// namespace component
}// namespace sofa

#endif //RigidScaleToAffineMultiMapping_INL
