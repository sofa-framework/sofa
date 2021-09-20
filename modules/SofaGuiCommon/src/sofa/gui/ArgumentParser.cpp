/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/ArgumentParser.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::gui
{

std::vector<std::string> ArgumentParser::extra = std::vector<std::string>();

ArgumentParser::ArgumentParser(int argc, char **argv)
    : m_options("SOFA")
{
    m_argc = argc;
    m_argv = argv;
    m_options.allow_unrecognised_options();
    m_options.add_options()("input-file", "input file", cxxopts::value<std::vector<std::string>>());
}

ArgumentParser::~ArgumentParser(){}

void ArgumentParser::addArgument(std::shared_ptr<cxxopts::Value> s, const std::string name, const std::string help)
{
    m_options.add_options()(name.c_str(), help.c_str(), s);
}

void ArgumentParser::addArgument(std::shared_ptr<cxxopts::Value> s, const std::string name, const std::string help, std::function<void(const ArgumentParser*, const std::string&)> callback)
{
    this->addArgument(s, name, help);
    m_mapCallbacks[name] = callback;
}

void ArgumentParser::addArgument(const std::string name, const std::string help)
{
    m_options.add_options()(name.c_str(), help.c_str());
}

void ArgumentParser::showHelp()
{
    std::cout << "This is a SOFA application. Here are the command line arguments" << std::endl;
    std::cout << m_options.help() << '\n';
}

void ArgumentParser::parse()
{
    std::vector<cxxopts::KeyValue> vecArg;
    try
    {
        m_options.parse_positional("input-file");
        auto temp = m_options.parse(m_argc, m_argv);
        vecArg = temp.arguments();
    }
    catch (const cxxopts::OptionException& e)
    {
        msg_error("ArgumentParser") << e.what();
        exit(EXIT_FAILURE);
    }

    // copy result
    for (const auto& arg : vecArg)
    {
        m_parseResult.insert({ arg.key(), arg.value() });

        //go through all possible keys (because of the short/long names)
        for (const auto& callback : m_mapCallbacks)
        {
            if (callback.first.find(arg.key()) != std::string::npos)
            {
                callback.second(this, arg.value());
                break;
            }
        }
    }
}

void ArgumentParser::showArgs()
{
    auto result = this->getMap();

    for (auto it = result.cbegin(); it != result.cend(); it++)
    {
        std::cout << "> " << it->first;
        if (it->second.empty()) {
            std::cout << "(empty)";
        }
        std::cout << "=" << it->second << std::endl;
    }
}

std::vector<std::string> ArgumentParser::getInputFileList()
{
    auto result = getMap();
    if (result.count("input-file"))
    {
        std::vector<std::string> tmp;
        cxxopts::values::parse_value(result["input-file"], tmp);
        return tmp;
    }
    return std::vector<std::string>();
}

const std::unordered_map<std::string, std::string>& ArgumentParser::getMap() const
{
    return m_parseResult;
}

} // namespace sofa::gui
