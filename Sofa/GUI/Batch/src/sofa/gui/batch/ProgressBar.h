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
#include <chrono>
#include <sofa/gui/batch/config.h>
#include <memory>

namespace indicators
{
    class ProgressBar;
    class IndeterminateProgressBar;
}

namespace sofa::gui::batch
{

class SOFA_GUI_BATCH_API ProgressBar
{
public:
    explicit ProgressBar(const int nbIterations);

    ~ProgressBar();

    void tick();

private:

    std::unique_ptr<indicators::ProgressBar> m_progressBar;
    std::unique_ptr<indicators::IndeterminateProgressBar> m_indeterminateProgressBar;

    int m_nbIterations{1};
    int m_currentNbIterations{};

    bool isDurationFromLastTickEnough() const;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_lastTick;
};

}
