/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/** \file Compliant test suite main file */
// Francois Faure,



#include <SofaTest/Sofa_test.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>

#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBoundaryCondition/ConstantForceField.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaMiscMapping/DistanceMapping.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <Compliant/numericalsolver/EigenSparseSolver.h>
#include <Compliant/numericalsolver/EigenSparseResponse.h>
#include <Compliant/compliance/UniformCompliance.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaRigid/RigidMapping.h>



namespace sofa
{
    typedef component::container::MechanicalObject<defaulttype::Vec1Types> MechanicalObject1;
    typedef component::mapping::RigidMapping<defaulttype::Rigid3Types, defaulttype::Vec3Types> RigidMappingRigid3_to_3;
    typedef component::mapping::SubsetMultiMapping<defaulttype::Vec3Types, defaulttype::Vec3Types> SubsetMultiMapping3_to_3;
    typedef component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> UniformMassRigid3;
    typedef component::mass::UniformMass<defaulttype::Vec1Types, SReal> UniformMass1;
    typedef component::mass::UniformMass<defaulttype::Vec3Types, SReal> UniformMass3;
    typedef component::projectiveconstraintset::FixedConstraint<defaulttype::Vec3Types> FixedConstraint3;

/** \page Page_CompliantTestSuite Compliant plugin test suite
 *
 * Class CompliantSolver_test provides helpers.
 *
 * Class Assembly_test checks the assembly of system matrices: mass, constraint Jacobian, etc.
 *
 * Class CompliantImplicitSolver_test checks the accuracy of the Implicit Euler integration in simple linear cases.
  */



/** Base class for tests of the Compliance plugin. Contains typedefs and helpers */
class CompliantSolver_test : public Sofa_test<>
{

public:


    typedef sofa::component::linearsolver::AssembledSystem::rmat SMatrix;

    typedef sofa::component::topology::EdgeSetTopologyContainer EdgeSetTopologyContainer;
    typedef sofa::defaulttype::Vec<3,SReal> Vec3;
    typedef sofa::component::forcefield::UniformCompliance<defaulttype::Vec1Types> UniformCompliance1;

    typedef modeling::MechanicalObject3 MechanicalObject3;

    // Vec3-Vec1
    typedef sofa::component::mapping::DistanceMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMapping31;
    typedef sofa::component::mapping::DistanceFromTargetMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceFromTargetMapping31;
    typedef sofa::component::mapping::DistanceMultiMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMultiMapping31;

    typedef sofa::component::interactionforcefield::StiffSpringForceField<defaulttype::Vec3Types>  StiffSpringForceField3;

protected:
    /** @name Helpers */
    ///@{

    /// Helper method to create strings used in various tests.
    simulation::Node::SPtr createCompliantString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, SReal totalMass, SReal complianceValue=0/*, SReal dampingRatio=0*/, bool isCompliant=true, SReal totalRestLength = -1 )
    {
        static unsigned numObject = 1;
        std::ostringstream oss;
        oss << "string_" << numObject++;
        SReal totalLength = totalRestLength<0 ? (endPoint-startPoint).norm() : totalRestLength;

        //--------
        simulation::Node::SPtr  string_node = parent->createChild(oss.str());

        MechanicalObject3::SPtr DOF = core::objectmodel::New<MechanicalObject3>();
        string_node->addObject(DOF);
        DOF->setName(oss.str()+"_DOF");

        UniformMass3::SPtr mass = core::objectmodel::New<UniformMass3>();
        string_node->addObject(mass);
        mass->setName(oss.str()+"_mass");
        mass->d_mass.setValue( totalMass/numParticles );




        //--------
        simulation::Node::SPtr extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

        MechanicalObject1::SPtr extensions = core::objectmodel::New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName(oss.str()+"_extensionsDOF");

        EdgeSetTopologyContainer::SPtr edgeSet = core::objectmodel::New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        DistanceMapping31::SPtr extensionMapping = core::objectmodel::New<DistanceMapping31>();
        extensionMapping->setName(oss.str()+"_extensionsMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );
        extension_node->addObject( extensionMapping );

        UniformCompliance1::SPtr compliance = core::objectmodel::New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->setName(oss.str()+"_extensionsCompliance");
        compliance->compliance.setValue(complianceValue);
        compliance->isCompliance.setValue(isCompliant);
        //        compliance->dampingRatio.setValue(dampingRatio);

        //--------
        // create the particles
        DOF->resize(numParticles);
        MechanicalObject3::WriteVecCoord x = DOF->writePositions();
        helper::vector<SReal> restLengths;
        for( unsigned i=0; i<numParticles; i++ )
        {
            SReal alpha = (SReal)i/(numParticles-1);
            x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
            if(i>0)
            {
                edgeSet->addEdge(i-1,i);
                restLengths.push_back( totalLength/(numParticles-1) );
            }
        }
        extensionMapping->f_restLengths.setValue( restLengths );

        return string_node;

    }

    /** Helper to create strings used in various tests.
      A struct with a constructor is more convenient than a method, because it allows us to subsequently access the nodes and components.
      */
    struct ParticleString
    {
        simulation::Node::SPtr  string_node; ///< root
        MechanicalObject3::SPtr DOF; ///< particle states
        UniformMass3::SPtr mass;

        simulation::Node::SPtr extension_node;
        MechanicalObject1::SPtr extensions;
        EdgeSetTopologyContainer::SPtr edgeSet;
        DistanceMapping31::SPtr extensionMapping;
        UniformCompliance1::SPtr compliance;

        ParticleString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, SReal totalMass )
        {
        static unsigned numObject = 1;
        std::ostringstream oss;
        oss << "string_" << numObject++;
        SReal totalLength = (endPoint-startPoint).norm();

        //--------
        string_node = parent->createChild(oss.str());
//        cerr<<"Particle string added as child of " << parent->getName() << endl;

        DOF = core::objectmodel::New<MechanicalObject3>();
        string_node->addObject(DOF);
        DOF->setName(oss.str()+"_DOF");

        mass = core::objectmodel::New<UniformMass3>();
        string_node->addObject(mass);
        mass->setName(oss.str()+"_mass");
        mass->d_mass.setValue( totalMass/numParticles );


        //--------
        extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

        extensions = core::objectmodel::New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName(oss.str()+"_extensionsDOF");

        edgeSet = core::objectmodel::New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        extensionMapping = core::objectmodel::New<DistanceMapping31>();
        extensionMapping->setName(oss.str()+"_extensionsMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );
        extension_node->addObject( extensionMapping );

        compliance = core::objectmodel::New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->setName(oss.str()+"_extensionsCompliance");


        //--------
        // create the particles
        DOF->resize(numParticles);
        MechanicalObject3::WriteVecCoord x = DOF->writePositions();
        helper::vector<SReal> restLengths;
        for( unsigned i=0; i<numParticles; i++ )
        {
            SReal alpha = (SReal)i/(numParticles-1);
            x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
            if(i>0)
            {
                edgeSet->addEdge(i-1,i);
                restLengths.push_back( totalLength/(numParticles-1) );
            }
        }
        extensionMapping->f_restLengths.setValue( restLengths );


        }

    };


    /// Return an identity matrix, or if not square, a matrix with 1 on each entry of the main diagonal
    static SMatrix makeSparseIdentity( unsigned rows, unsigned cols )
    {
        SMatrix m(rows,cols);
        for(unsigned i=0; i<rows; i++ )
        {
            if(i<cols)
            {
                m.startVec(i);
                m.insertBack(i,i) = 1.0;
            }
        }
        m.finalize();
        return m;
    }

    /// Return an identity matrix, or if not square, a matrix with 1 on each entry of the main diagonal
    static modeling::DenseMatrix makeIdentity( unsigned rows, unsigned cols )
    {
        modeling::DenseMatrix m(rows,cols);
        for(unsigned i=0; i<rows; i++ )
        {
            m(i,i) = 1.0;
        }
        return m;
    }

    /// Return true if the matrices have same size and all their entries are equal within the given tolerance. Specialization on Eigen matrices.
    static bool matricesAreEqual( const modeling::DenseMatrix m1, const SMatrix& sm2, SReal tolerance=100*std::numeric_limits<SReal>::epsilon() )
    {
        modeling::DenseMatrix m2 = sm2;
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        modeling::DenseMatrix diff = m1 - m2;
        bool areEqual = std::abs(diff.maxCoeff())<tolerance && std::abs(diff.minCoeff())<tolerance;
        if( !areEqual )
        {
            std::cerr<<"CompliantSolver_test::matricesAreEqual1, tolerance = "<< tolerance << ", difference = " << std::endl << diff << std::endl;
        }
        return areEqual;

    }

    /// Return true if the matrices have same size and all their entries are equal within the given tolerance. Specialization on Eigen matrices.
    static bool matricesAreEqual( const SMatrix m1, const SMatrix& m2, SReal tolerance=100*std::numeric_limits<SReal>::epsilon() )
    {
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        SMatrix diff = m1 - m2;
        for (int k=0; k<diff.outerSize(); ++k)
            for (SMatrix::InnerIterator it(diff,k); it; ++it)
            {
                if( fabs(it.value()) >tolerance )
                {
                    std::cerr<<"CompliantSolver_test::matricesAreEqual2, tolerance = "<< tolerance << ", difference = " << std::endl << it.value() << std::endl;
                    return false;
                }

            }
        return true;

    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool vectorsAreEqual( const modeling::Vector& m1, const modeling::Vector& m2, SReal tolerance=100*std::numeric_limits<SReal>::epsilon() )
    {
        if( m1.size()!=m2.size() )
        {
            std::cerr<<"CompliantSolver_test::vectorsAreEqual: sizes " << m1.size() << " != " << m2.size() << std::endl;
            return false;
        }

        modeling::Vector diff = m1-m2;
        bool areEqual = std::abs(diff.maxCoeff())<tolerance && std::abs(diff.minCoeff())<tolerance;
        if( !areEqual )
        {
            std::cerr<<"CompliantSolver_test::vectorsAreEqual, tolerance = "<< tolerance << ", difference = " << std::endl << diff << std::endl;
        }
        return areEqual;
    }


    ///@}




};

}//sofa
