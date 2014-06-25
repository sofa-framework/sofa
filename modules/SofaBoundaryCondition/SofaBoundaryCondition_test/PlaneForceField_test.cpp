/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "Sofa_test.h"
#include <SofaBoundaryCondition/PlaneForceField.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

template <typename _DataTypes>
struct PlaneForceField_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
	typedef sofa::component::forcefield::PlaneForceField<DataTypes> PlaneForceFieldType;
	typename PlaneForceFieldType::SPtr planeForceFieldSPtr;

	bool test()
	{
		//Init
		planeForceFieldSPtr = sofa::core::objectmodel::New<PlaneForceFieldType>();
		planeForceFieldSPtr->setStiffness(500);

		// Test if the stiffness value is set correctly
		if(planeForceFieldSPtr->stiffness.getValue()!=500)
        {  
           ADD_FAILURE() << "Error stiffness expected: " << 500 << endl <<
                             " actual " << planeForceFieldSPtr->stiffness.getValue() << endl;
           return false;   
        }
		return true;
	}

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(PlaneForceField_test, DataTypes);
// first test case
TYPED_TEST( PlaneForceField_test , testValue )
{
    ASSERT_TRUE(  this->test() );
}
}// namespace sofa







