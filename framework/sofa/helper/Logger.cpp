/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "Logger.h"

#include <sofa/helper/system/console.h>


namespace sofa
{

namespace helper
{

Logger::SPtr Logger::s_mainLogger = Logger::SPtr(new TTYLogger());

Logger::Logger(): m_currentLevel(Logger::All)
{
}

Logger::~Logger()
{
}

void Logger::setLevel(Level level)
{
    m_currentLevel = level;
}

Logger::Level Logger::getLevel()
{
    return m_currentLevel;
}

Logger& Logger::getMainLogger()
{
    return *s_mainLogger.get();
}

void Logger::setMainLogger(boost::shared_ptr<Logger> logger)
{
    s_mainLogger = logger;
}


TTYLogger::TTYLogger()
{
    m_prefixes[Logger::Debug] = "[Debug] ";
    m_prefixes[Logger::Info] = "[Info] ";
    m_prefixes[Logger::Warning] = "[Warning] ";
    m_prefixes[Logger::Error] = "[Error] ";
    m_colors[Logger::Debug] = Console::DEFAULT_COLOR;
    m_colors[Logger::Info] = Console::BRIGHT_GREEN;
    m_colors[Logger::Warning] = Console::RED;
    m_colors[Logger::Error] = Console::BRIGHT_RED;
}

void TTYLogger::log(Level level, const std::string& message, const std::string& location)
{
    if (level < getLevel())
        return;

    std::ostream& output = (level >= Warning? std::cerr: std::cout);

    if (m_prefixes[level].size() > 0)
        output << m_colors[level] << m_prefixes[level] << Console::DEFAULT_COLOR;
    if (location.size() > 0)
        output << Console::YELLOW << "[" << location << "] " << Console::DEFAULT_COLOR;
    output << message << std::endl;
}


} // namespace helper

} // namespace sofa
