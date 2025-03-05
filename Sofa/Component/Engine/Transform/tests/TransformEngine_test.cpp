/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/component/engine/transform/TransformEngine.h>
#include <sofa/type/Quat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

using sofa::component::engine::transform::TransformEngine;

sofa::type::Vec3		nullptr_VEC(0,0,0);
sofa::type::Vec3		nullptr_SCALE(1,1,1);

sofa::type::Vec3		INPUT_POS(1,0,0);
sofa::type::Quat<SReal>		INPUT_QUAT(0,0,0,1);

sofa::type::Vec3		TRANSLATION(1,2,3);
sofa::type::Vec3		OUTPUT_TRANSLATION_POS(2,2,3);

sofa::type::Vec3		ROTATION(0,0,90);
sofa::type::Vec3		OUTPUT_ROTATION_POS(0,1,0);
sofa::type::Quat<SReal>		OUTPUT_ROTATION_QUAT(0,0,0.7071,0.7071);

sofa::type::Vec3		SCALE(5,10,20);
sofa::type::Vec3		OUTPUT_SCALE_POS(5,0,0);

sofa::type::Vec3		OUTPUT_ROTATION_SCALE_POS(0,5,0);


namespace sofa 
{

using type::Vec3;

template <typename _DataTypes>
class TransformEngine_test : public ::testing::Test, public TransformEngine<_DataTypes>
{
public:
    typedef _DataTypes DataTypes;
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Real Real;
	typedef sofa::type::Quat<SReal> Quat;


	TransformEngine_test(bool init = true)
	{
		if (init)
			TransformEngine<DataTypes>::init();
		this->f_outputX.cleanDirty();
		this->f_inputX.cleanDirty();
		this->cleanDirty();
	}

	/*****
	 * Helpers
	 */
	VecCoord initInputVecCoord()
	{
		VecCoord testVec;
		testVec.push_back(sofa::testing::createCoord<DataTypes>(INPUT_POS, INPUT_QUAT));
	
		return testVec;
	}

	void setInputTransformation(Vec3 translation, Vec3 rotation, Vec3 scale)
	{
		this->f_inputX.setValue(initInputVecCoord()); 

		this->translation.setValue( translation );
		this->rotation.setValue( rotation );
		this->scale.setValue( scale );
	}

	void testOutput(Vec3 pos, Quat quat = Quat())
	{
		Coord output = this->f_outputX.getValue()[0];
		Coord referenceCoord = sofa::testing::createCoord<DataTypes>(pos, quat);

		const Real diff			= (referenceCoord - output).norm();
		const Real abs_error	= 1e-5;
		if (diff > abs_error) 
			ADD_FAILURE() << "There is a difference between the output:\n" << output << "\nand the expected value:\n" << referenceCoord;
	}

	void testInput(core::objectmodel::BaseData* data)
	{
		data->setDirtyValue();
		ASSERT_TRUE( this->f_outputX.isDirty() );
		this->f_outputX.cleanDirty();
		this->cleanDirty();
	}


};

template <typename _DataTypes>
class TransformEngine_test_uninitialized : public TransformEngine_test < _DataTypes >
{
public:
	TransformEngine_test_uninitialized()
        : TransformEngine_test<_DataTypes>(false)
	{
	}
};

namespace
{

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    defaulttype::Vec1Types,
    defaulttype::Vec2Types,
    defaulttype::Vec3Types,
    defaulttype::Rigid2Types,
    defaulttype::Rigid3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(TransformEngine_test, DataTypes);
TYPED_TEST_SUITE(TransformEngine_test_uninitialized, DataTypes);

// test dirty flag on inputs, uninitialized
TYPED_TEST( TransformEngine_test_uninitialized , uninitialized )
{
	this->testInput(&this->f_inputX);
	this->testInput(&this->translation);
	this->testInput(&this->rotation);
	this->testInput(&this->scale);
}

// test dirty flag on inputs
TYPED_TEST( TransformEngine_test , input )
{
    this->testInput(&this->f_inputX);
	this->testInput(&this->translation);
	this->testInput(&this->rotation);
	this->testInput(&this->scale);
}

// test translation
TYPED_TEST( TransformEngine_test , translation )
{
    this->setInputTransformation(TRANSLATION, nullptr_VEC, nullptr_SCALE);
    this->testOutput(OUTPUT_TRANSLATION_POS);
}

// test rotation
TYPED_TEST( TransformEngine_test , rotation )
{
    this->setInputTransformation(nullptr_VEC, ROTATION, nullptr_SCALE);
    this->testOutput(OUTPUT_ROTATION_POS, OUTPUT_ROTATION_QUAT);
}

// test scale
TYPED_TEST( TransformEngine_test , scale )
{
    this->setInputTransformation(nullptr_VEC, nullptr_VEC, SCALE);
    this->testOutput(OUTPUT_SCALE_POS);
}

// test translation-rotation composite
TYPED_TEST( TransformEngine_test , translationRotation )
{
    this->setInputTransformation(TRANSLATION, ROTATION, nullptr_SCALE);
    this->testOutput(OUTPUT_ROTATION_POS + TRANSLATION, OUTPUT_ROTATION_QUAT);
}

// test translation-scale composite
TYPED_TEST( TransformEngine_test , translationScale )
{
    this->setInputTransformation( TRANSLATION, nullptr_VEC, SCALE );
    this->testOutput(OUTPUT_SCALE_POS + TRANSLATION);
}

// test rotation-scale composite
TYPED_TEST( TransformEngine_test , rotationScale )
{
    this->setInputTransformation( nullptr_VEC, ROTATION, SCALE );
    this->testOutput(OUTPUT_ROTATION_SCALE_POS, OUTPUT_ROTATION_QUAT);
}

// test translation-rotation-scale composite
TYPED_TEST( TransformEngine_test , translationRotationScale )
{
    this->setInputTransformation( TRANSLATION, ROTATION, SCALE );
    this->testOutput(OUTPUT_ROTATION_SCALE_POS + TRANSLATION, OUTPUT_ROTATION_QUAT);
}

}// namespace

}// namespace sofa
