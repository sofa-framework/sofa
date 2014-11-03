#include <plugins/SofaTest/Sofa_test.h>
#include <plugins/SceneCreator/SceneCreator.h>
#include <assert.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/component/loader/MeshObjLoader.h>
#include <sofa/component/loader/MeshObjLoader.h>
#include <sofa/component/engine/GenerateRigidMass.h>
#include <sofa/simulation/common/Node.h>
#include <plugins/Registration/InertiaAlign.h>
//#include <projects/GenerateRigid/GenerateRigid.h>
namespace sofa{
using namespace modeling;


    struct InertiaAlign_test : public Sofa_test<>
    {
        typedef component::InertiaAlign InertiaAlign;
        typedef component::loader::MeshObjLoader MeshObjLoader;
        typedef component::engine::GenerateRigidMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3Mass> GenerateRigidMass;


        bool translation_test()
        {
            bool res = true;
            SReal epsilon = 0.00001;
            //root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "InertiaAlign.scn");
            helper::io::Mesh meshSource;
            MeshObjLoader::SPtr meshLoaderSource = New<MeshObjLoader>();
            meshLoaderSource->m_filename.setValue("/home/pierre/Workspace/boxes.obj");
            //TODO : changer ces chemins et potentiellement le modele
            meshLoaderSource->load();
            meshSource.Create("/home/pierre/Workspace/boxes.obj");
            helper::io::Mesh meshTarget;
            MeshObjLoader::SPtr meshLoaderTarget = New<MeshObjLoader>();
            meshLoaderTarget->m_filename.setValue("/home/pierre/Workspace/boxes.obj");
            meshLoaderTarget->load();
            meshTarget.Create("/home/pierre/Workspace/boxes.obj");

            /// Seed for random value
            long seed = 7;

            /// Random generator
            helper::RandomGenerator randomGenerator;
            randomGenerator.initSeed(seed);

            SReal translation_x,translation_y,translation_z;
            translation_x = randomGenerator.random<SReal>(-10.0,10.0);
            translation_y = randomGenerator.random<SReal>(-10.0,10.0);
            translation_z = randomGenerator.random<SReal>(-10.0,10.0);

            SReal rotation_x,rotation_y,rotation_z;
            rotation_x = randomGenerator.random<SReal>(-90.0,90.0);
            rotation_y = randomGenerator.random<SReal>(-90.0,90.0);
            rotation_z = randomGenerator.random<SReal>(-90.0,90.0);

            /// Compute Inertia Matrix
            meshLoaderSource->applyTranslation(translation_x, translation_y, translation_z);
            meshLoaderSource->applyRotation(rotation_x, rotation_y, rotation_z);
            defaulttype::Rigid3Mass massSource;
            defaulttype::Vec3d centerSource ;

            //GenerateRigid (massSource, centerSource, &meshSource);

            defaulttype::Vec3d centerTarget ;
            sofa::defaulttype::Vector3 translation;
            defaulttype::Rigid3Mass massTarget;

            GenerateRigidMass::SPtr rigidSource = New<GenerateRigidMass>();
            rigidSource->m_positions.setValue(meshLoaderSource->positions);
            rigidSource->m_triangles.setValue(meshLoaderSource->triangles);
            rigidSource->init();
            rigidSource->update();


            GenerateRigidMass::SPtr rigidTarget = New<GenerateRigidMass>();
            rigidTarget->m_positions.setValue(meshSource.getVertices());
            rigidTarget->init();
            rigidTarget->update();

            InertiaAlign::SPtr Ia = New<InertiaAlign>();
            //translatedPositions, centerTarget , centerSource, &translation ,massSource.inertiaMatrix,massTarget.inertiaMatrix );
;
            Ia->targetC.setValue(centerTarget);
            Ia->sourceC.setValue(centerSource);
            Ia->translation.setValue(translation);
            Ia->m_positions.setValue(meshLoaderSource.position);
            Ia->m_positiont.setValue(meshLoaderTarget.position);

            //Ia->targetInertiaMatrix.setValue(rigidSource.inertiaMatrix.getValue());
            //Ia->sourceInertiaMatrix.setValue(rigidTarget.inertiaMatrix.getValue());
            Ia->init();
            for (unsigned int i=0; i<translatedPositions.size();i++)
            {
                SReal diff = abs(translatedPositions[i][0] - meshSource.getVertices()[i][0]);
                if(diff>epsilon)
                    res = false;
                diff = abs(translatedPositions[i][1] - meshSource.getVertices()[i][1]);
                if(diff>epsilon)
                    res = false;
                diff = abs(translatedPositions[i][2] - meshSource.getVertices()[i][2]);
                if(diff>epsilon)
                    res = false;
            }
            return res;
        }

    };
    TEST_F(InertiaAlign_test, InertiaAlign){    ASSERT_TRUE(translation_test());  }

}


