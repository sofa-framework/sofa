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
/** \file Compliant test suite main file */
// Francois Faure,

/** \page Page_CompliantTestSuite Compliant plugin test suite
  */


#include <plugins/SofaTest/Solver_test.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>

#include <sofa/component/init.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/forcefield/ConstantForceField.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/ExtensionMapping.h>
#include <sofa/component/mapping/DistanceMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>

#include "../odesolver/AssembledSolver.h"
#include "../numericalsolver/LDLTSolver.h"
#include "../compliance/UniformCompliance.h"

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/CompareState.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/system/atomic.h>


#include <Eigen/Dense>
using std::cout;

using namespace sofa;
using namespace sofa::component;
using namespace modeling;
using sofa::helper::vector;


/** Base class for tests of the Compliance plugin. Contains typedefs and helpers */
class CompliantSolver_test : public Solver_test
{
protected:
    typedef linearsolver::AssembledSystem::rmat SMatrix;

    typedef component::topology::EdgeSetTopologyContainer EdgeSetTopologyContainer;
    typedef defaulttype::Vec<3,SReal> Vec3;
    typedef forcefield::UniformCompliance<Vec1Types> UniformCompliance1;

    // Vec3-Vec1
    typedef mapping::ExtensionMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> ExtensionMapping31;
    typedef mapping::DistanceMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMapping31;


protected:
    /** @name Helpers */
    ///@{

    /// Helper method to create strings used in various tests.
    Node::SPtr createCompliantString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass, double complianceValue=0/*, double dampingRatio=0*/, bool isCompliant=true, SReal totalRestLength = -1 )
    {
        static unsigned numObject = 1;
        std::ostringstream oss;
        oss << "string_" << numObject++;
        SReal totalLength = totalRestLength<0 ? (endPoint-startPoint).norm() : totalRestLength;

        //--------
        simulation::Node::SPtr  string_node = parent->createChild(oss.str());

        MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
        string_node->addObject(DOF);
        DOF->setName(oss.str()+"_DOF");

        UniformMass3::SPtr mass = New<UniformMass3>();
        string_node->addObject(mass);
        mass->setName(oss.str()+"_mass");
        mass->mass.setValue( totalMass/numParticles );




        //--------
        simulation::Node::SPtr extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName(oss.str()+"_extensionsDOF");

        EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setName(oss.str()+"_extensionsMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );
        extension_node->addObject( extensionMapping );

        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
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
            double alpha = (double)i/(numParticles-1);
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
        ExtensionMapping31::SPtr extensionMapping;
        UniformCompliance1::SPtr compliance;

        ParticleString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass )
        {
        static unsigned numObject = 1;
        std::ostringstream oss;
        oss << "string_" << numObject++;
        SReal totalLength = (endPoint-startPoint).norm();

        //--------
        string_node = parent->createChild(oss.str());
//        cerr<<"Particle string added as child of " << parent->getName() << endl;

        DOF = New<MechanicalObject3>();
        string_node->addObject(DOF);
        DOF->setName(oss.str()+"_DOF");

        mass = New<UniformMass3>();
        string_node->addObject(mass);
        mass->setName(oss.str()+"_mass");
        mass->mass.setValue( totalMass/numParticles );


        //--------
        extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

        extensions = New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName(oss.str()+"_extensionsDOF");

        edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setName(oss.str()+"_extensionsMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );
        extension_node->addObject( extensionMapping );

        compliance = New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->setName(oss.str()+"_extensionsCompliance");


        //--------
        // create the particles
        DOF->resize(numParticles);
        MechanicalObject3::WriteVecCoord x = DOF->writePositions();
        helper::vector<SReal> restLengths;
        for( unsigned i=0; i<numParticles; i++ )
        {
            double alpha = (double)i/(numParticles-1);
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
    static DenseMatrix makeIdentity( unsigned rows, unsigned cols )
    {
        DenseMatrix m(rows,cols);
        for(unsigned i=0; i<rows; i++ )
        {
            m(i,i) = 1.0;
        }
        return m;
    }

    /// Return true if the matrices have same size and all their entries are equal within the given tolerance. Specialization on Eigen matrices.
    static bool matricesAreEqual( const DenseMatrix m1, const SMatrix& sm2, SReal tolerance=100*std::numeric_limits<SReal>::epsilon() )
    {
        DenseMatrix m2 = sm2;
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        DenseMatrix diff = m1 - m2;
        bool areEqual = abs(diff.maxCoeff()<tolerance && abs(diff.minCoeff()<tolerance));
        if( !areEqual )
        {
            cerr<<"CompliantSolver_test::matricesAreEqual1, tolerance = "<< tolerance << ", difference = " << endl << diff << endl;
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
                    cerr<<"CompliantSolver_test::matricesAreEqual2, tolerance = "<< tolerance << ", difference = " << endl << it.value() << endl;
                    return false;
                }

            }
        return true;

    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool vectorsAreEqual( const Vector& m1, const Vector& m2, SReal tolerance=100*std::numeric_limits<SReal>::epsilon() )
    {
        if( m1.size()!=m2.size() )
        {
            cerr<<"CompliantSolver_test::vectorsAreEqual: sizes " << m1.size() << " != " << m2.size() << endl;
            return false;
        }

        Vector diff = m1-m2;
        bool areEqual = abs(diff.maxCoeff()<tolerance && abs(diff.minCoeff()<tolerance));
        if( !areEqual )
        {
            cerr<<"CompliantSolver_test::vectorsAreEqual, tolerance = "<< tolerance << ", difference = " << endl << diff << endl;
        }
        return areEqual;
    }


    ///@}




};

