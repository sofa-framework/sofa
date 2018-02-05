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
//========================================================
// Yet another command line parser.
// Francois Faure, iMAGIS-GRAVIR, May 2001
//========================================================

#ifndef SOFA_HELPER_ARGUMENTPARSER_H
#define SOFA_HELPER_ARGUMENTPARSER_H

#include <iostream>
#include <string>

#include <sofa/helper/helper.h>
#include <sofa/helper/logging/Messaging.h>

#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>

namespace po = boost::program_options;


namespace sofa
{

namespace helper
{

/** Command line parser

This object parses arguments from a command line or from an input stream.
@see Argument
*/


class SOFA_HELPER_API ArgumentParser
{


public:

    /**
     *****************************************************************************************
     *  @brief      constructor
     *     *
     *  @param      argc from command line
     *  @param      argv from command line
     ****************************************************************************************/

    ArgumentParser(int argc, char **argv);
    ~ArgumentParser();

    /**
     *****************************************************************************************
     *  @brief      addArgument
     *
     *  @usage      Can be call if you want to add parameters
     *
     *  @param      s boost_program_option semantic for examples @see runSofa/main.cpp
     *  @param      name argument name such as "help,h" after the comma it's a character use as short name
     *  @param      help describing the argument
     ****************************************************************************************/

    void addArgument(const po::value_semantic* s, const std::string name, const std::string help);
    /**
     *****************************************************************************************
     *  @brief      addArgument
     *
     *  @usage      Can be call if you want to add parameters
     *
     *  @param      name argument name such as "help,h" after the comma it's a character use as short name
     *  @param      help describing the argument
     ****************************************************************************************/
    void addArgument(const std::string name, const std::string help);

    /// simply display the help (You need to add -h --help options in your main and call this function by yourself @see runSofa/main.cpp)
    void showHelp();
    /// this is the main function. You have to call this function if you want to parse the arguments given to the constructor
    void parse();
    /// display args with values
    void showArgs();

    po::variables_map getVariableMap();
    std::vector<std::string> getInputFileList();

    /** last parsed extra arguments */
    static std::vector<std::string> extra; ///< extra parameter needed for python (arguments)
    /// return extra_args needed for python (arguments)
    static const std::vector<std::string> extra_args() { return extra; }

private:
    int m_argc; ///< simple argc parameter copied from constructor
    char **m_argv; ///< simple argv parameter copied from constructor
    po::variables_map vm; ///< Variable map containing the variable name with its value obtained from parse
    po::options_description desc; ///< desc contains every options you want to parse. Each options has argument name, help, variables ref ...
    po::positional_options_description positional_option; ///< this is used for parsing input files without any parameters



};




} // namespace helper

} // namespace sofa

#endif
