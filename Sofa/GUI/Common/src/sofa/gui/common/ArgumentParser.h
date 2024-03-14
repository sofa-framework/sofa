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
#pragma once

#include <sofa/gui/common/config.h>

#include <sofa/helper/logging/Messaging.h>

#include <iostream>
#include <string>

#include <cxxopts.hpp>

namespace sofa::gui::common
{

/** Command line parser

This object parses arguments from a command line or from an input stream.
@see Argument
*/


class SOFA_GUI_COMMON_API ArgumentParser
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
     *  @param      s a sptr to a cxxopts::Value for examples @see runSofa/main.cpp
     *  @param      name argument name such as "help,h" after the comma it's a character use as short name
     *  @param      help describing the argument
     ****************************************************************************************/
    void addArgument(std::shared_ptr<cxxopts::Value> s, const std::string name, const std::string help);
    /**
     *****************************************************************************************
     *  @brief      addArgument
     *
     *  @usage      Can be call if you want to add parameters
     *
     *  @param      s a sptr to a cxxopts::Value for examples @see runSofa/main.cpp
     *  @param      name argument name such as "h,help" after the comma it's a character use as short name
     *  @param      help describing the argument
     *  @param      callback will be called when parsing is done and this arg has been modified
     ****************************************************************************************/
    void addArgument(std::shared_ptr<cxxopts::Value> s, const std::string name, const std::string help, std::function<void(const ArgumentParser*, const std::string&)> callback);
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
    /// get the value associated to the key (if this key exists or the type is correct)
    template <typename T>
    bool getValueFromKey(const std::string& key, T& value) const
    {
        bool ret = false;
        auto result = this->getMap();
        try
        {
            if (result.count(key))
            {
                cxxopts::values::parse_value(result[key], value);
                ret = true;
            }
        }
        catch (const std::bad_cast& e1) // could not cast the value to T
        {
            SOFA_UNUSED(e1);
            ret = false;
        }
        catch (const cxxopts::exceptions::exception& e2) // option is not present, etc
        {
            SOFA_UNUSED(e2);
            ret = false;
        }

        return ret;
    }

    const std::unordered_map<std::string, std::string>& getMap() const;
    std::vector<std::string> getInputFileList();


    /** last parsed extra arguments */
    static std::vector<std::string> extra; ///< extra parameter needed for python (arguments)
    /// return extra_args needed for python (arguments)
    static const std::vector<std::string> extra_args() { return extra; }

private:
    int m_argc; ///< simple argc parameter copied from constructor
    char **m_argv; ///< simple argv parameter copied from constructor

    cxxopts::Options m_options;
    std::map<std::string, std::function<void(const ArgumentParser*, const std::string&)>> m_mapCallbacks;
    std::unordered_map<std::string, std::string> m_parseResult;
};

} // namespace sofa::gui::common
