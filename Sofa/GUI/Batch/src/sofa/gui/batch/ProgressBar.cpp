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
#include "indicators/indicators.hpp"
#include <csignal>

namespace sofa::gui::batch
{
ProgressBar::ProgressBar(const int nbIterations)
    : m_nbIterations(nbIterations)
    , m_currentNbIterations(1)
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
    m_lastTick = std::chrono::high_resolution_clock::now();
}

ProgressBar::~ProgressBar()
{
    indicators::show_console_cursor(true);
}

bool ProgressBar::isDurationFromLastTickEnough() const
{
    const auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> durationFromLastTick = now - m_lastTick;
    constexpr auto minimumDuration = std::chrono::milliseconds(1000 / 30); // 30Hz
    return durationFromLastTick < minimumDuration;
}

void ProgressBar::tick()
{
    //checking that enough time has passed to update the progress bar
    //otherwise it could slow down the simulation, printing too often in the console
    if (m_currentNbIterations < m_nbIterations - 1 && isDurationFromLastTickEnough())
    {
        ++m_currentNbIterations;
        return;
    }

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

    m_lastTick = std::chrono::high_resolution_clock::now();
    ++m_currentNbIterations;
}

}
