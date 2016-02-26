/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
******************************************************************************/

#include <iostream>
#include <string>
#include <regex>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace fs = boost::filesystem;


enum LICENSE_TYPE {
    LGPL = 0,
    GPL = 1};

enum SOFTWARE_TYPE {
        LIBRARY = 0,
        PROGRAM = 1};



/**
 * @brief main
 * @param argc
 * @param argv : argv[1] has to be the sofa root
 * @return
 */
int main(int argc, char** argv)
{
    // We're gonna update the file with the following extensions : h, cpp, inl
    const std::regex files("\\.(h|cpp|inl|c|cu|h\\.in)$");
    // the regex we use to find the old license. The first line must have 70+ asterisks, the second line has to contain SOFA, it is 17 lines long
    const std::regex licenseRegex("( )*/(\\*){70,}\n\\*( ){5,}(SOFA)(.*\n){17}");

    const std::string license_LGPL =
            "/******************************************************************************\n"
            "*       SOFA, Simulation Open-Framework Architecture, development version     *\n"
            "*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *\n"
            "*                                                                             *\n"
            "* This library is free software; you can redistribute it and/or modify it     *\n"
            "* under the terms of the GNU Lesser General Public License as published by    *\n"
            "* the Free Software Foundation; either version 2.1 of the License, or (at     *\n"
            "* your option) any later version.                                             *\n"
            "*                                                                             *\n"
            "* This library is distributed in the hope that it will be useful, but WITHOUT *\n"
            "* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *\n"
            "* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *\n"
            "* for more details.                                                           *\n"
            "*                                                                             *\n"
            "* You should have received a copy of the GNU Lesser General Public License    *\n"
            "* along with this library; if not, write to the Free Software Foundation,     *\n"
            "* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *\n"
            "*******************************************************************************\n";

    const std::string license_GPL =
            "/******************************************************************************\n"
            "*       SOFA, Simulation Open-Framework Architecture, development version     *\n"
            "*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *\n"
            "*                                                                             *\n"
            "* This library is free software; you can redistribute it and/or modify it     *\n"
            "* under the terms of the GNU General Public License as published by the Free  *\n"
            "* Software Foundation; either version 2 of the License, or (at your option)   *\n"
            "* any later version.                                                          *\n"
            "*                                                                             *\n"
            "* This library is distributed in the hope that it will be useful, but WITHOUT *\n"
            "* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *\n"
            "* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *\n"
            "* more details.                                                               *\n"
            "*                                                                             *\n"
            "* You should have received a copy of the GNU General Public License along     *\n"
            "* with this library; if not, write to the Free Software Foundation, Inc., 51  *\n"
            "* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *\n"
            "*******************************************************************************\n";

    const std::string license_LGPL_program = boost::algorithm::replace_all_copy(license_LGPL, "library", "program");
    const std::string license_GPL_program = boost::algorithm::replace_all_copy(license_GPL, "library", "program");

    if(argc > 1)
    {
        const fs::path dir(argv[1]);
        std::cout << "updating licence in directory : " << dir.string() << std::endl;
        if (fs::exists(dir) && fs::is_directory(dir))
        {
            for (fs::recursive_directory_iterator dirIT(dir), end ; dirIT != end ; ++dirIT)
            {
                // if the filename ends with .h, .inl, .c, .cu, .h.in or .cpp
                if (fs::is_regular_file(dirIT->status()) && std::regex_search(dirIT->path().filename().string(), files))
                {
                    std::fstream stream(fs::absolute(dirIT->path()).string(), std::ios::in | std::ios::out);
                    std::stringstream buffer;
                    buffer << stream.rdbuf();

                    const std::string& text = buffer.str();
                    std::smatch match;
                    if(std::regex_search(text, match, licenseRegex))
                    {
                        const std::string& matchSTR = match.str(0);
                        stream.clear();
                        stream.seekp(0);

                        std::string const * license = nullptr;
                        LICENSE_TYPE license_t = LICENSE_TYPE::GPL;
                        SOFTWARE_TYPE software_t = SOFTWARE_TYPE::LIBRARY;

                        if (std::regex_search(matchSTR, std::regex("lesser", std::regex::ECMAScript | std::regex::icase)))
                        {
                            license_t = LICENSE_TYPE::LGPL;
                        }
                        if (std::regex_search(matchSTR, std::regex("program", std::regex::ECMAScript | std::regex::icase)))
                        {
                            software_t = SOFTWARE_TYPE::PROGRAM;
                        }

                        if (license_t == LICENSE_TYPE::LGPL)
                        {
                            if (software_t == SOFTWARE_TYPE::LIBRARY)
                            {
                                license = &license_LGPL;
                            } else {
                                license = &license_LGPL_program;
                            }
                        } else { // GPL
                            if (software_t == SOFTWARE_TYPE::LIBRARY)
                            {
                                license = &license_GPL;
                            } else {
                                license = &license_GPL_program;
                            }
                        }

                        std::string newFileStr = std::regex_replace(text, licenseRegex, *license);
                        newFileStr.pop_back();
                        stream << newFileStr;
                    }
                }
            }
        } else {
            std::cerr << "The given directory doesn't exist : " << dir.string() << std::endl;
        }
    }
    return 0;
}




