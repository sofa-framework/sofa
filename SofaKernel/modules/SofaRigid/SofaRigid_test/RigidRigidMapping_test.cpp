/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Mapping_test.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaRigid/RigidRigidMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {
namespace {
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;


/**  Test suite for RigidRigidMapping.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _RigidRigidMapping>
struct RigidRigidMappingTest : public Mapping_test<_RigidRigidMapping>
{

    typedef _RigidRigidMapping RigidRigidMapping;
    typedef Mapping_test<RigidRigidMapping> Inherit;

    typedef typename RigidRigidMapping::In In;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::Coord::Pos InCoordPos;
    typedef container::MechanicalObject<In> InMechanicalObject;
    typedef typename InMechanicalObject::ReadVecCoord  ReadInVecCoord;
    typedef typename InMechanicalObject::WriteVecCoord WriteInVecCoord;
    typedef typename InMechanicalObject::WriteVecDeriv WriteInVecDeriv;
    typedef typename InCoord::Pos Translation;
    typedef typename InCoord::Rot Rot;
    typedef typename In::Real InReal;
    typedef Mat<In::spatial_dimensions,In::spatial_dimensions,InReal> RotationMatrix;


    typedef typename RigidRigidMapping::Out Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::Coord::Pos OutCoordPos;
    typedef container::MechanicalObject<Out> OutMechanicalObject;
    typedef typename OutMechanicalObject::WriteVecCoord WriteOutVecCoord;
    typedef typename OutMechanicalObject::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutMechanicalObject::ReadVecCoord ReadOutVecCoord;
    typedef typename OutMechanicalObject::ReadVecDeriv ReadOutVecDeriv;

    RigidRigidMapping* rigidRigidMapping;

    RigidRigidMappingTest()
    {
        rigidRigidMapping = static_cast<RigidRigidMapping*>( this->mapping );
        // RigidRigidMapping assembly is not implemented
        // you can use Compliant AssembledRigidRigidMapping for that purpose
        this->flags &= ~Inherit::TEST_ASSEMBLY_API;

        this->errorFactorDJ = 10;
    }

    OutVecCoord create_childCoord()
    {
        OutVecCoord xout(1);
        Out::set( xout[0] ,10.,0.,0.);
        Rot rot = Rot( OutCoordPos(0,0,1), M_PI/2 );
        Out::setCRot( xout[0], rot );
        return xout;
    }

    InVecCoord create_initial_parentCoord()
    {
        const int Nin=5;
        InVecCoord xin_init(Nin);
        Rot rot = Rot( OutCoordPos(0,0,1), M_PI/4 );
        for( int i=0; i<Nin; i++ )
        {
            OutCoordPos x(-40+10*i,0,0);
            xin_init[i].getCenter()      = OutCoordPos(-10,0,0) + rot.rotate(x);
            xin_init[i].getOrientation() = rot;
        }
        return xin_init;
    }

    InVecCoord create_final_parentCoord()
    {
        const int Nin=5;
        InVecCoord xin(Nin);
        for( int i=0; i<Nin; i++ )
        {
            OutCoordPos x(-40+10*i,0,0);
            Rot rot( OutCoordPos(0,0,1), M_PI/2 - i*M_PI/((Nin-1)*2)  ); // rotation around z different for every particle to test index
            xin[i].getCenter()      = OutCoordPos(-10,0,0) + rot.rotate(x);
            xin[i].getOrientation() = rot;

        }
        return xin;
    }

    bool test_translation()
    {
        // Parent
        InVecCoord xin_init(1);
        In::set( xin_init[0] ,1.,0.,0.);
        Rot rot1 = Rot( InCoordPos(0,0,1), 0 );
        In::setCRot(xin_init[0],rot1);

        // Children
        OutVecCoord xout(1);
        Out::set( xout[0] ,2.,0.,0.);
        Rot rot2 = Rot( OutCoordPos(0,0,1), 0 );
        Out::setCRot( xout[0], rot2 );

        rigidRigidMapping->globalToLocalCoords.setValue(false); // initial child positions are given in local coordinates

        // Translate parent
        InVecCoord xin(1);
        double tx = 2; double ty = 0; double tz = 0;
        double alpha = M_PI/2;
        In::set( xin[0] ,tx,ty,tz);
        Rot rot3 = Rot( InCoordPos(0,0,1), alpha);
        In::setCRot(xin[0],rot3);

        // Expected child coords
        OutVecCoord expectedChildCoords(1);
        //Center
        defaulttype::Vector3 translationVector (tx,ty,tz);
        expectedChildCoords[0].getCenter() = rot3.rotate(xout[0].getCenter()) + translationVector;
        // Orientation
        expectedChildCoords[0].getOrientation() = xin[0].getOrientation();

        // test
        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    bool test_no_index_no_movement_worldCoords()
    {
        OutVecCoord xout = create_childCoord();
        InVecCoord xin_init = create_initial_parentCoord();

        const int Nin=xin_init.size(), Nout=xout.size();
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        rigidRigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates

        InVecCoord xin=xin_init;
        OutVecCoord expectedChildCoords = xout;
        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    bool test_with_index_no_movement_worldCoords()
    {
        OutVecCoord xout = create_childCoord();
        InVecCoord xin_init = create_initial_parentCoord();

        const int Nin=xin_init.size(), Nout=xout.size();
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        rigidRigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates
        rigidRigidMapping->index.setValue( 1 );

        InVecCoord xin=xin_init;
        OutVecCoord expectedChildCoords = xout;
        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    bool test_no_index_worldCoords()
    {
        OutVecCoord xout = create_childCoord();
        InVecCoord xin_init = create_initial_parentCoord();

        const int Nin=xin_init.size(), Nout=xout.size();
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        rigidRigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates

        InVecCoord xin=create_final_parentCoord();

        int index = 0;
        RotationMatrix Rfinal, Rinit, invRinit; // matrices are a unified model for 3D (quaternion) and 2D (scalar).
        xin[index].writeRotationMatrix(Rfinal);
        xin_init[index].writeRotationMatrix(Rinit);
        invRinit = Rinit.transposed();
        OutVecCoord expectedChildCoords(Nout);
        expectedChildCoords[0].getCenter()      = xin[index].getCenter()  +  Rfinal * (invRinit*(xout[0].getCenter() - xin_init[index].getCenter()));
        expectedChildCoords[0].getOrientation() = xin_init[index].getOrientation().inverse() * xin[index].getOrientation() * xout[0].getOrientation();

        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    bool test_index_0_worldCoords()
    {
        OutVecCoord xout = create_childCoord();
        InVecCoord xin_init = create_initial_parentCoord();

        const int Nin=xin_init.size(), Nout=xout.size();
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        int index = 0;
        rigidRigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates
        rigidRigidMapping->index.setValue(index);

        InVecCoord xin=create_final_parentCoord();


        RotationMatrix Rfinal, Rinit, invRinit; // matrices are a unified model for 3D (quaternion) and 2D (scalar).
        xin[index].writeRotationMatrix(Rfinal);
        xin_init[index].writeRotationMatrix(Rinit);
        invRinit = Rinit.transposed();
        OutVecCoord expectedChildCoords(Nout);
        expectedChildCoords[0].getCenter()      = xin[index].getCenter()  +  Rfinal * (invRinit*(xout[0].getCenter() - xin_init[index].getCenter()));
        expectedChildCoords[0].getOrientation() = xin_init[index].getOrientation().inverse() * xin[index].getOrientation() * xout[0].getOrientation();

        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }

    bool test_index_2_worldCoords()
    {
        OutVecCoord xout = create_childCoord();
        InVecCoord xin_init = create_initial_parentCoord();

        const int Nin=xin_init.size(), Nout=xout.size();
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        int index = 2;
        rigidRigidMapping->globalToLocalCoords.setValue(true); // initial child positions are given in world coordinates
        rigidRigidMapping->index.setValue(index);

        InVecCoord xin=create_final_parentCoord();


        RotationMatrix Rfinal, Rinit, invRinit; // matrices are a unified model for 3D (quaternion) and 2D (scalar).
        xin[index].writeRotationMatrix(Rfinal);
        xin_init[index].writeRotationMatrix(Rinit);
        invRinit = Rinit.transposed();
        OutVecCoord expectedChildCoords(Nout);
        expectedChildCoords[0].getCenter()      = xin[index].getCenter()  +  Rfinal * (invRinit*(xout[0].getCenter() - xin_init[index].getCenter()));
        expectedChildCoords[0].getOrientation() = xin_init[index].getOrientation().inverse() * xin[index].getOrientation() * xout[0].getOrientation();

        return this->runTest(xin_init,xout,xin,expectedChildCoords);
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<mapping::RigidRigidMapping<defaulttype::Rigid3Types,defaulttype::Rigid3Types> > DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(RigidRigidMappingTest, DataTypes);

TYPED_TEST( RigidRigidMappingTest , test_translation )
{
    this->errorMax = 10;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_translation());
}

TYPED_TEST( RigidRigidMappingTest , no_index_no_movement_worldCoords )
{
    this->errorMax = 200;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_no_index_no_movement_worldCoords());
}


TYPED_TEST( RigidRigidMappingTest , with_index_no_movement_worldCoords )
{
    this->errorMax = 200;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_with_index_no_movement_worldCoords());
}

TYPED_TEST( RigidRigidMappingTest , no_index_worldCoords )
{
    this->errorMax = 200;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_no_index_worldCoords());
}

TYPED_TEST( RigidRigidMappingTest , index_0_worldCoords )
{
    this->errorMax = 200;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_index_0_worldCoords());
}

TYPED_TEST( RigidRigidMappingTest , index_2_worldCoords )
{
    this->errorMax = 200;
    this->deltaRange.second = this->errorMax*100;
    ASSERT_TRUE(this->test_index_2_worldCoords());
}




}//anonymous namespace
} // namespace sofa
