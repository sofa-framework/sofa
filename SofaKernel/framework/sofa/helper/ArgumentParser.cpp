/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "ArgumentParser.h"
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

std::vector<std::string> ArgumentParser::extra = std::vector<std::string>();

ArgumentParser::ArgumentParser(int argc, char **argv){
    m_argc = argc;
    m_argv = argv;
    positional_option.add("input-file", -1);
    desc.add_options()("input-file", po::value<std::vector<std::string> >(), "input file");
}

ArgumentParser::~ArgumentParser(){}

void ArgumentParser::addArgument(const po::value_semantic* s, const std::string name, const std::string help)
{
    desc.add_options()(name.c_str(), s, help.c_str());
}

void ArgumentParser::addArgument(const std::string name, const std::string help)
{
    desc.add_options()(name.c_str(), help.c_str());
}

void ArgumentParser::showHelp()
{
    std::cout << "This is a SOFA application. Here are the command line arguments" << std::endl;
    std::cout << desc << '\n';
}

void ArgumentParser::parse()
{
    try {
        po::store(po::command_line_parser(m_argc, m_argv).options(desc).positional(positional_option).run(), vm);

        if (vm.find("argv") != vm.end())
            extra = vm.at("argv").as<std::vector<std::string> >();
    }
    catch (po::error const& e) {
        std::cerr << e.what() << '\n';
        exit( EXIT_FAILURE );
    }
    po::notify(vm);

}

void ArgumentParser::showArgs()
{
    for (po::variables_map::iterator it = vm.begin(); it != vm.end(); it++) {
        std::cout << "> " << it->first;
        if (((boost::any)it->second.value()).empty()) {
            std::cout << "(empty)";
        }
        if (vm[it->first].defaulted() || it->second.defaulted()) {
            std::cout << "(default)";
        }
        std::cout << "=";

        bool is_char;
        try {
            boost::any_cast<const char *>(it->second.value());
            is_char = true;
        } catch (const boost::bad_any_cast &) {
            is_char = false;
        }
        bool is_str;
        try {
            boost::any_cast<std::string>(it->second.value());
            is_str = true;
        } catch (const boost::bad_any_cast &) {
            is_str = false;
        }

        if (((boost::any)it->second.value()).type() == typeid(int)) {
            std::cout << vm[it->first].as<int>() << std::endl;
        } else if (((boost::any)it->second.value()).type() == typeid(bool)) {
            std::cout << vm[it->first].as<bool>() << std::endl;
        } else if (((boost::any)it->second.value()).type() == typeid(unsigned int)) {
            std::cout << vm[it->first].as<unsigned int>() << std::endl;
        } else if (((boost::any)it->second.value()).type() == typeid(double)) {
            std::cout << vm[it->first].as<double>() << std::endl;
        } else if (is_char) {
            std::cout << vm[it->first].as<const char * >() << std::endl;
        } else if (is_str) {
            std::string temp = vm[it->first].as<std::string>();
            if (temp.size()) {
                std::cout << temp << std::endl;
            } else {
                std::cout << "true" << std::endl;
            }
        } else { // Assumes that the only remainder is vector<string>
            try {
                std::vector<std::string> vect = vm[it->first].as<std::vector<std::string> >();
                unsigned int i = 0;
                for (std::vector<std::string>::iterator oit=vect.begin();
                     oit != vect.end(); oit++, ++i) {
                    std::cout << "\r> " << it->first << "[" << i << "]=" << (*oit) << std::endl;
                }
            } catch (const boost::bad_any_cast &) {
                std::cout << "UnknownType(" << ((boost::any)it->second.value()).type().name() << ")" << std::endl;
            }
        }
    }
}

std::vector<std::string> ArgumentParser::getInputFileList()
{
    if (vm.find("input-file") != vm.end())
        return vm.at("input-file").as<std::vector<std::string> >();
    return std::vector<std::string>();
}

po::variables_map ArgumentParser::getVariableMap()
{
    return vm;
}

} // namespace helper

} // namespace sofa
