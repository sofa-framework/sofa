#include "RegressionScene_test.h"


namespace sofa 
{

std::string RegressionScene_test::getTestName(const testing::TestParamInfo<RegressionSceneTest_Data>& p)
{
    const std::string& path = p.param.fileScenePath;
    std::size_t pos = path.find_last_of("/"); // get name of the file without path
    
    if (pos != std::string::npos)
        pos++;

    std::string name = path.substr(pos);
    name = name.substr(0, name.find_last_of(".")); // get name of the file without extension

    return name;
}


void RegressionScene_test::runRegressionStateTest(RegressionSceneTest_Data data)
{
    msg_info("Regression_test") << "  Testing " << data.fileScenePath;

    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    simulation::Simulation* simulation = simulation::getSimulation();

    // Load the scene
    sofa::simulation::Node::SPtr root = simulation->load(data.fileScenePath.c_str());

    simulation->init(root.get());

    // TODO lancer visiteur pour dumper MO
    // comparer ce dump avec le fichier sceneName.regressionreference

    bool initializing = false;

    if (helper::system::FileSystem::exists(data.fileRefPath) && !helper::system::FileSystem::isDirectory(data.fileRefPath))
    {
        // Add CompareState components: as it derives from the ReadState, we use the ReadStateActivator to enable them.
        sofa::component::misc::CompareStateCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
        //            compareVisitor.setCreateInMapping(true);
        compareVisitor.setSceneName(data.fileRefPath);
        compareVisitor.execute(root.get());

        sofa::component::misc::ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
        v_read.execute(root.get());
    }
    else // create reference
    {
        msg_warning("Regression_test") << "Non existing reference created: " << data.fileRefPath;

        // just to create an empty file to know it is already init
        std::ofstream filestream(data.fileRefPath.c_str());
        filestream.close();

        initializing = true;
        sofa::component::misc::WriteStateCreator writeVisitor(sofa::core::ExecParams::defaultInstance());
        //            writeVisitor.setCreateInMapping(true);
        writeVisitor.setSceneName(data.fileRefPath);
        writeVisitor.execute(root.get());

        sofa::component::misc::WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
        v_write.execute(root.get());
    }

    for (unsigned int i = 0; i<data.steps; ++i)
    {
        simulation->animate(root.get(), root->getDt());
    }

    if (!initializing)
    {
        // Read the final error: the summation of all the error made at each time step
        sofa::component::misc::CompareStateResult result(sofa::core::ExecParams::defaultInstance());
        result.execute(root.get());

        double errorByDof = result.getErrorByDof() / double(result.getNumCompareState());
        if (errorByDof > data.epsilon)
        {
            msg_error("Regression_test") << data.fileScenePath << ":" << msgendl
                << "    TOTALERROR: " << result.getTotalError() << msgendl
                << "    ERRORBYDOF: " << errorByDof;
        }
    }

    // Clear and prepare for next scene
    simulation->unload(root.get());
    root.reset();
}


/// performing the regression test on every plugins/projects

INSTANTIATE_TEST_CASE_P(regression,
    RegressionScene_test,
    ::testing::ValuesIn(regressionState_tests.m_listScenes),
    RegressionScene_test::getTestName);

TEST_P(RegressionScene_test, all_tests) { runRegressionStateTest(GetParam()); }


} // namespace sofa
