/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
The arguments are described using a pointer, a short name, a long name and a help message. Mandatory arguments are declared using method "parameter", optional arguments are declared using method "option".
Once all arguments declared, operator () does the parsing.
The special option -h or --help displays help on all arguments.
See examples argumentParserLine_test.cpp and argumentParserFile_test.cpp
@see Argument
*/


class SOFA_HELPER_API ArgumentParser
{


public:

    ArgumentParser(int a, char *b[]);
    ~ArgumentParser();

    void addArgument(const po::value_semantic* s, const std::string name, const std::string help);
    void addArgument(const std::string name, const std::string help);
    void showHelp();
    void parse();
    void showArgs();

    po::variables_map getVariableMap();
    std::vector<std::string> getInputFileList();

    /** last parsed extra arguments */
    static std::vector<std::string> extra;
    static const std::vector<std::string> extra_args() { return extra; }

private:
    int argc;
    char **argv;
    po::variables_map vm;
    po::options_description desc;
    po::positional_options_description p;



};




} // namespace helper

} // namespace sofa

#endif
