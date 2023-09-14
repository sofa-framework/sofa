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
#include <sofa/gui/batch/ProgressBar.h>
#include "indicators.hpp"
#include <csignal>

namespace sofa::gui::batch
{
ProgressBar::ProgressBar(const int nbIterations): m_nbIterations(nbIterations)
{
    if (nbIterations != -1)
    {
        m_progressBar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"\r["},
            indicators::option::Fill{"#"},
            indicators::option::Lead{"#"},
            indicators::option::Remainder{"-"},
            indicators::option::End{"]"},
            indicators::option::PostfixText{},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
        );
    }
    else
    {
        m_indeterminateProgressBar = std::make_unique<indicators::IndeterminateProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"\r["},
            indicators::option::Fill{"."},
            indicators::option::Lead{"<==>"},
            indicators::option::End{"]"},
            indicators::option::PostfixText{},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
        );
    }

    indicators::show_console_cursor(false);
}

ProgressBar::~ProgressBar()
{
    if (m_progressBar)
    {
        m_progressBar->mark_as_completed();
    }
    indicators::show_console_cursor(true);
}

void ProgressBar::tick()
{
    if (m_nbIterations != -1 && m_progressBar)
    {
        m_progressBar->set_option(indicators::option::PostfixText{std::to_string(m_currentNbIterations) + "/" + std::to_string(m_nbIterations)});
        m_progressBar->set_progress(100 * m_currentNbIterations / m_nbIterations);
    }
    else if (m_nbIterations == -1 && m_indeterminateProgressBar)
    {
        m_indeterminateProgressBar->tick();
        m_indeterminateProgressBar->set_option(indicators::option::PostfixText{std::to_string(m_currentNbIterations)});
    }
    ++m_currentNbIterations;
}

}
