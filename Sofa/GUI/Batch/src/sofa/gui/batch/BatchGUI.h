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
#pragma once

#include <sofa/gui/batch/config.h>
#include <sofa/gui/common/BaseGUI.h>
#include <sofa/simulation/fwd.h>
#include <string>
#include <sstream>

namespace sofa::gui::common
{
    class ArgumentParser;
}

namespace sofa::gui::batch
{

class SOFA_GUI_BATCH_API BatchGUI : public common::BaseGUI
{

public:

    /// @name methods each GUI must implement
    /// @{

    BatchGUI();

    void setScene(sofa::simulation::NodeSPtr groot, const char* filename="", bool temporaryFile=false) override;

    void resetScene();

    int mainLoop() override;
    void redraw() override;
    int closeGUI() override;

    sofa::simulation::Node* currentSimulation() override;

    /// @}

    /// @name registration of each GUI
    /// @{

    static BaseGUI* CreateGUI(const char* name, sofa::simulation::NodeSPtr groot = nullptr, const char* filename = nullptr);
    static int RegisterGUIParameters(common::ArgumentParser* argumentParser);
    static void OnNbIterChange(const common::ArgumentParser*, const std::string& strValue);


    static const signed int DEFAULT_NUMBER_OF_ITERATIONS;
    /// @}

    bool canBeDefaultGUI() const override { return false; }

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~BatchGUI() override;

    void startDumpVisitor();
    void stopDumpVisitor();

    std::ostringstream m_dumpVisitorStream;

    sofa::simulation::NodeSPtr groot;
    std::string filename;
    static signed int nbIter;
    static std::string nbIterInp;
    inline static bool hideProgressBar { false };

    /// Return true if the timer output string has a json string and the timer is setup to output json
    static bool canExportJson(const std::string& timerOutputStr, const std::string& timerId);

    /// Export a text file (with json extension) containing the timer output string
    void exportJson(const std::string& timerOutputStr, int iterationNumber) const;
};

} // namespace sofa::gui::batch
