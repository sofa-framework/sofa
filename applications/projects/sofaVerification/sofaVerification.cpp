#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <ctime>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

void apply(std::vector< std::string> &files, unsigned int iterations, bool reinit)
{
    using namespace sofa::helper::system;
    sofa::simulation::tree::GNode* groot = NULL;
    for (unsigned int i=0; i<files.size(); ++i)
    {
        groot = dynamic_cast< sofa::simulation::tree::GNode* >( sofa::simulation::tree::getSimulation()->load(files[i].c_str()));
        if (groot == NULL)
        {
            std::cerr << "CANNOT open " << files[i] << " !\n";
            continue;
        }


        std::string dir = SetDirectory::GetParentDir(SetDirectory::GetParentDir(DataRepository.getFile("Sofa.ini").c_str()).c_str());
        std::string file = SetDirectory::GetFileName(files[i].c_str());
        file = dir + std::string("/applications/projects/sofaVerification/simulation/") + file;
        if (reinit)
        {
            sofa::simulation::WriteStateCreator compareVisitor;
            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(file);
            compareVisitor.execute(groot);

            sofa::simulation::WriteStateActivator v_write(true);
            v_write.execute(groot);
        }
        else
        {
            sofa::simulation::CompareStateCreator compareVisitor;
            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(file);
            compareVisitor.execute(groot);

            sofa::simulation::ReadStateActivator v_read(true);
            v_read.execute(groot);
        }
        clock_t curtime = clock();
        std::cout << "Computing " <<  iterations << " for " << files[i] <<  std::endl;
        for (unsigned int i=0; i<iterations; i++)
        {
            sofa::simulation::tree::getSimulation()->animate(groot);
        }
        double t = (clock() - curtime)/((double)CLOCKS_PER_SEC);

        std::cout << iterations << " iterations done in " << t  << " seconds" <<std::endl;

        if (!reinit)
        {
            sofa::simulation::CompareStateResult result;
            result.execute(groot);
            std::cout << "ERROR : " << result.getError() << "\n";

        }
        sofa::simulation::tree::getSimulation()->unload(groot);
    }
}

int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    std::string fileName ;
    std::vector<std::string> files;
    unsigned int iterations=100;
    bool reinit=false;




    sofa::helper::parse(&files, "This is a SOFA verification. Here are the command line arguments")
    .option(&reinit,'r',"reinit","Recreate the references state files")
    .option(&iterations, 'i',"iteration", "Number of iterations for testing")
    (argc,argv);


    if (!files.empty()) fileName = files[0];

    sofa::helper::system::DataRepository.findFile(fileName);


    //Get the list of scenes to test
    files.clear();
    std::ifstream end(fileName.c_str());
    std::string s;
    while( end >> s )
    {
        sofa::helper::system::DataRepository.findFile(s);
        files.push_back(s);
    }
    end.close();


    if (reinit) apply(files, iterations, true);
    else
        apply(files, iterations, false);


    return 0;
}
