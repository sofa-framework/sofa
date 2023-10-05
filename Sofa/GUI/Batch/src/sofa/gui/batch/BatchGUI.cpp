/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gui/batch/BatchGUI.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/common/ArgumentParser.h>

#include <cxxopts.hpp>

#include <fstream>
#include <string>
#include <iomanip>
#include <sofa/gui/batch/ProgressBar.h>


namespace sofa::gui::batch
{

using sofa::helper::AdvancedTimer;
using namespace sofa::gui::common;

constexpr signed int BatchGUI::DEFAULT_NUMBER_OF_ITERATIONS = 1000;
signed int BatchGUI::nbIter = BatchGUI::DEFAULT_NUMBER_OF_ITERATIONS;
std::string BatchGUI::nbIterInp="";
BatchGUI::BatchGUI()
    : groot(nullptr)
{
}

BatchGUI::~BatchGUI()
{
}

int BatchGUI::mainLoop()
{
    if (groot)
    {   
        if (nbIter != -1)
        {   
            msg_info("BatchGUI") << "Computing " << nbIter << " iterations." << msgendl;
        }
        else
        {
            msg_info("BatchGUI") << "Computing infinite iterations." << msgendl;
        }

        AdvancedTimer::begin("Animate");
        sofa::simulation::node::animate(groot.get());
        msg_info("BatchGUI") << "Processing." << AdvancedTimer::end("Animate", groot->getTime(), groot->getDt()) << msgendl;
        const sofa::simulation::Visitor::ctime_t rtfreq = sofa::helper::system::thread::CTime::getRefTicksPerSec();
        const sofa::simulation::Visitor::ctime_t tfreq = sofa::helper::system::thread::CTime::getTicksPerSec();
        sofa::simulation::Visitor::ctime_t rt = sofa::helper::system::thread::CTime::getRefTime();
        sofa::simulation::Visitor::ctime_t t = sofa::helper::system::thread::CTime::getFastTime();
          
        signed int i = 1; //one simulation step is animated above

        std::unique_ptr<ProgressBar> progressBar;
        if (!hideProgressBar)
        {
            progressBar = std::make_unique<ProgressBar>(nbIter);
        }

        while (i <= nbIter || nbIter == -1)
        {
            if (i != nbIter)
            {
                AdvancedTimer::begin("Animate");

                sofa::simulation::node::animate(groot.get());

                const std::string timerOutputStr = AdvancedTimer::end("Animate", groot->getTime(), groot->getDt());
                if (canExportJson(timerOutputStr, "Animate"))
                {
                    exportJson(timerOutputStr, i);
                }
            }

            if ( i == nbIter || (nbIter == -1 && i%1000 == 0) )
            {
                t = sofa::helper::system::thread::CTime::getFastTime()-t;
                rt = sofa::helper::system::thread::CTime::getRefTime()-rt;

                msg_info("BatchGUI") << i << " iterations done in " << ((double)t)/((double)tfreq) << " s ( " << (((double)tfreq)*i)/((double)t) << " FPS)." << msgendl;
                msg_info("BatchGUI") << i << " iterations done in " << ((double)rt)/((double)rtfreq) << " s ( " << (((double)rtfreq)*i)/((double)rt) << " FPS)." << msgendl;
                
                if (nbIter == -1) // Additional message for infinite iterations
                {
                     msg_info("BatchGUI") << "Press Ctrl + C (linux)/ Command + period (mac) to stop " << msgendl;
                }
            }

            if (progressBar)
            {
                progressBar->tick();
            }

            i++;
        }
    }
    return 0;
}

void BatchGUI::redraw()
{
}

int BatchGUI::closeGUI()
{
    delete this;
    return 0;
}

void BatchGUI::setScene(sofa::simulation::Node::SPtr groot, const char* filename, bool )
{
    this->groot = groot;
    this->filename = (filename?filename:"");

    resetScene();
}


void BatchGUI::resetScene()
{
    sofa::simulation::Node* root = currentSimulation();

    if ( root )
    {
        root->setTime(0.);
        sofa::simulation::node::reset(root);

        sofa::simulation::UpdateSimulationContextVisitor(sofa::core::execparams::defaultInstance()).execute(root);
    }
}

void BatchGUI::startDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Node* root = currentSimulation();
    if (root)
    {
        m_dumpVisitorStream.str("");
        sofa::simulation::Visitor::startDumpVisitor(&m_dumpVisitorStream, root->getTime());
    }
#endif
}

void BatchGUI::stopDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::stopDumpVisitor();
    m_dumpVisitorStream.flush();
    m_dumpVisitorStream.str("");
#endif
}

sofa::simulation::Node* BatchGUI::currentSimulation()
{
    return groot.get();
}

BaseGUI* BatchGUI::CreateGUI(const char* name, sofa::simulation::Node::SPtr groot, const char* filename)
{
    BatchGUI::mGuiName = name;
    BatchGUI* gui = new BatchGUI();
    gui->setScene(groot, filename);
    return gui;
}

int BatchGUI::RegisterGUIParameters(ArgumentParser* argumentParser)
{
    argumentParser->addArgument(
        cxxopts::value<std::string>(nbIterInp),
        "n,nbIter",
        "(only batch) Number of iterations of the simulation",
        BatchGUI::OnNbIterChange
    );
    argumentParser->addArgument(
        cxxopts::value<bool>(hideProgressBar)->default_value("false"),
        "hideProgressBar",
        "if defined, hides the progress bar"
    );
    return 0;
}

void BatchGUI::OnNbIterChange(const ArgumentParser* argumentParser, const std::string& strValue)
{
    SOFA_UNUSED(argumentParser);

    nbIterInp = strValue;
    const size_t inpLen = nbIterInp.length();

    if (nbIterInp == "infinite")
    {
        nbIter = -1;
    }
    else if (inpLen)
    {
        nbIter = std::stoi(nbIterInp);
    }
    else
    {
        nbIter = DEFAULT_NUMBER_OF_ITERATIONS;
    }
}

bool BatchGUI::canExportJson(const std::string& timerOutputStr, const std::string& timerId)
{
    const auto outputType = AdvancedTimer::getOutputType(AdvancedTimer::IdTimer(timerId));
    if (outputType == AdvancedTimer::outputType::JSON || outputType == AdvancedTimer::outputType::LJSON)
    {
        //timerOutputStr is not empty when the AdvancedTimer has been setup with an interval (AdvancedTimer::setInterval)
        //and the number of iterations is reached
        return !timerOutputStr.empty() && timerOutputStr != "null";
    }
    return false;
}

void BatchGUI::exportJson(const std::string &timerOutputStr, int iterationNumber) const
{
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << iterationNumber;

    const std::string jsonFilename =
            sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename.c_str()) + "_" + ss.str() + ".json";
    msg_info("BatchGUI") << "Writing " << jsonFilename;
    std::ofstream out(jsonFilename);
    out << timerOutputStr;
    out.close();
}

} // namespace sofa::gui::batch
