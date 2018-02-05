/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

/* Francois Faure, 2013 */
#include <SofaTest/Mapping_test.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {
  namespace {
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;


/**  Test suite for RigidMapping.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _RigidMapping>
struct RigidMappingTest : public Mapping_test<_RigidMapping>
{

    typedef _RigidMapping RigidMapping;
    typedef Mapping_test<RigidMapping> Inherit;

    typedef typename RigidMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef container::MechanicalObject<InDataTypes> InMechanicalObject;
    typedef typename InMechanicalObject::ReadVecCoord  ReadInVecCoord;
    typedef typename InMechanicalObject::WriteVecCoord WriteInVecCoord;
    typedef typename InMechanicalObject::WriteVecDeriv WriteInVecDeriv;
    typedef typename InCoord::Pos Translation;
    typedef typename InCoord::Rot Rotation;
    typedef typename InDataTypes::Real InReal;
    typedef Mat<InDataTypes::spatial_dimensions,InDataTypes::spatial_dimensions,InReal> RotationMatrix;


    typedef typename RigidMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef container::MechanicalObject<OutDataTypes> OutMechanicalObject;
    typedef typename OutMechanicalObject::WriteVecCoord WriteOutVecCoord;
    typedef typename OutMechanicalObject::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutMechanicalObject::ReadVecCoord ReadOutVecCoord;
    typedef typename OutMechanicalObject::ReadVecDeriv ReadOutVecDeriv;


    RigidMapping* rigidMapping;

    RigidMappingTest()
    {
        this->errorFactorDJ = 200;

        rigidMapping = static_cast<RigidMapping*>( this->mapping );

        if( InDataTypes::spatial_dimensions != 3 )
        {
            // RigidMapping::getK is not yet implemented for 2D rigids
            this->flags &= ~Inherit::TEST_getK;
        }
    }


    /** @name Test_Cases
      For each of these cases, we can test if the mapping work
      */
    ///@{
    /** One frame, with particles given in local coordinates.
     * This tests the mapping from local to world coordinates.
    */
    bool test_oneRigid_fourParticles_localCoords()
    {
        const int Nin=1, Nout=4;
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        // child positions
        rigidMapping->globalToLocalCoords.setValue(false); // initial child positions are given in local coordinates
        rigidMapping->geometricStiffness.setValue(1); // full unsymmetrized geometric stiffness
        OutVecCoord xout(Nout);
        // vertices of the unit tetrahedron
        OutDataTypes::set( xout[0] ,0.,0.,0.);
        OutDataTypes::set( xout[1] ,1.,0.,0.);
        OutDataTypes::set( xout[2] ,0.,1.,0.);
        OutDataTypes::set( xout[3] ,0.,0.,1.);

        // parent position
        InVecCoord xin(Nin);
        InDataTypes::set( xin[0], 1.,-2.,3. );
        Rotation rot = InDataTypes::rotationEuler(-1.,2.,-3.);
        InDataTypes::setCRot( xin[0], rot );


        // expected mapped values
        OutVecCoord expectedChildCoords(Nout);
        RotationMatrix m;
        xin[0].writeRotationMatrix(m);
        for(unsigned i=0; i<xout.size(); i++ )
        {
            // note that before init, xout is still in relative coordinates
            expectedChildCoords[i] = xin[0].getCenter() + m * xout[i];
        }

        // The same xin is used twice since xout is given in local coordinates while expectedChildCoords is given in world coordinates
        // Here we simply test the mapping from local to world coordinates
        return this->runTest(xin,xout,xin,expectedChildCoords);
    }

    /** One frame, with particles given in world coordinates.
     * This requires the mapping from world to local coordinates.
    */
    bool test_oneRigid_fourParticles_worldCoords()
    {
        const int Nin=1, Nout=4;
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        // child positions
        rigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates
        rigidMapping->geometricStiffness.setValue(1); // full unsymmetrized geometric stiffness
        OutVecCoord xout(Nout);
        // vertices of the unit tetrahedron
        OutDataTypes::set( xout[0] ,0.,0.,0.);
        OutDataTypes::set( xout[1] ,1.,0.,0.);
        OutDataTypes::set( xout[2] ,0.,1.,0.);
        OutDataTypes::set( xout[3] ,0.,0.,1.);

        // initial parent position
        InVecCoord xin_init(Nin);
        InDataTypes::set( xin_init[0], -3.,1.,-2. );
        Rotation rot_init = InDataTypes::rotationEuler(3.,-1.,2.);
        InDataTypes::setCRot( xin_init[0], rot_init );

        // final parent position
        InVecCoord xin(Nin);
        InDataTypes::set( xin[0], 1.,-2.,3. );
        Rotation rot = InDataTypes::rotationEuler(-1.,2.,-3.);
        InDataTypes::setCRot( xin[0], rot );

        // expected mapped values
        OutVecCoord expectedChildCoords(Nout);
        RotationMatrix Rfinal, Rinit, invRinit; // matrices are a unified model for 3D (quaternion) and 2D (scalar).
        xin[0].writeRotationMatrix(Rfinal);
        xin_init[0].writeRotationMatrix(Rinit);
        invRinit = Rinit.transposed();
        for(unsigned i=0; i<xout.size(); i++ )
        {
            // transformation from initial to final parent position: Tfinal.Rfinal.Rinit^{-1}.Tinit^{-1}
            expectedChildCoords[i] = xin[0].getCenter()  +  Rfinal * (invRinit*(xout[i] - xin_init[0].getCenter()));
        }

        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    /// @todo test with several frames


    ///@}


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
mapping::RigidMapping<defaulttype::Rigid2Types,defaulttype::Vec2Types>,
mapping::RigidMapping<defaulttype::Rigid3Types,defaulttype::Vec3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(RigidMappingTest, DataTypes);
// first test case
TYPED_TEST( RigidMappingTest , oneRigid_fourParticles_localCoords )
{
    // child coordinates given directly in parent frame
    ASSERT_TRUE(this->test_oneRigid_fourParticles_localCoords());
}
TYPED_TEST( RigidMappingTest , oneRigid_fourParticles_worldCoords )
{
    // child coordinates given in word frame
    this->errorMax = 100.; // a larger error occurs, probably due to the world to local mapping at init:
    ASSERT_TRUE(this->test_oneRigid_fourParticles_worldCoords());
}

}//anonymous namespace
} // namespace sofa
