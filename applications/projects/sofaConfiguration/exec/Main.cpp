
/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include <iostream>
#include <fstream>
#include <stdlib.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <direct.h>
#include <shellapi.h>
#endif
#if defined (__APPLE__)
#include <sys/param.h>
#include <mach-o/dyld.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

#include "SofaConfiguration.h"

#include <vector>
#include <set>
#include <algorithm>


#include <qapplication.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

using sofa::gui::qt::DEFINES;
using sofa::gui::qt::CONDITION;
using sofa::gui::qt::TYPE_CONDITION;
using sofa::gui::qt::OPTION;
using sofa::gui::qt::ARCHI;

void removeInitialCharacter(std::string &s, char c)
{
    unsigned int i=0;
    for (; i<s.size(); ++i)
    {
        if (s[i] != c) break;
    }
    s=s.substr(i);
}

void removeFinalCharacter(std::string &s, char c)
{
    int i=s.size()-1;
    for (; i>=0; --i)
    {
        if (s[i] != c) break;
    }
    s.resize(i+1);
}

void removeComment(std::string &s)
{
    std::size_t found=s.find('#');
    if (found != std::string::npos) s.resize(found);
}

std::string currentCategory;


struct classcompare
{
    bool operator() (const DEFINES& a, const DEFINES& b) const
    {return a.name < b.name;}
};

void processDescription(std::string &description, std::size_t pos)
{
    description=description.substr(pos+10);
}

void processOption(std::string &name, bool &activated, std::size_t pos)
{
    std::string line=name;
    removeInitialCharacter(line,' ');
    if (line[0] == '#') activated=false;
    else                activated=true;

    name = name.substr(pos+10);
    removeInitialCharacter(name,' ');
    removeComment(name);
    removeFinalCharacter(name,' ');
}

void processTextOption(std::string &description, std::string &name, bool &activated, std::size_t pos)
{
    removeInitialCharacter(description,' ');
    if (description[0] == '#')
    {
        activated=false;
        description=description.substr(1);
    }
    else                activated=true;

    name=description;
    name.resize(pos+1);
    removeInitialCharacter(name,' ');
    removeFinalCharacter(name,' ');

    description = description.substr(pos+1);
    removeInitialCharacter(description,' ');
    removeFinalCharacter(description,' ');
}

void processCondition(std::string &description, bool &presence, TYPE_CONDITION &type, std::size_t pos)
{
    std::size_t posContains=description.find("contains(");
    std::size_t boolNot=description.find('!');
    if (posContains!=std::string::npos)
    {
        type=OPTION;
        std::size_t separator=description.find(',');
        std::string type=description;  type.resize(separator); type=type.substr(posContains+9);
        std::string option=description.substr(separator+1);
        separator=option.find(')'); option.resize(separator);
        if (type=="DEFINES") description=option;

    }
    else
    {
        type=ARCHI;
        description.resize(pos);
    }

    if (boolNot == std::string::npos || boolNot > pos) presence=true;
    else presence=false;
}

void processCategory(std::string &description)
{
    removeInitialCharacter(description,'#');
    removeInitialCharacter(description,' ');
    removeFinalCharacter(description,'#');
    removeFinalCharacter(description,' ');
}

void parse(std::ifstream &in, std::vector<DEFINES>  &listOptions)
{


    enum State {NONE, CATEGORY};
    int STATE=NONE;

    std::string description;
    std::vector< CONDITION > conditions;
    std::string text;
    while (std::getline(in, text))
    {
        removeInitialCharacter(text,' ');
        std::size_t found;
        //Three keywords: Uncomment, DEFINES +=, contains(
        switch (STATE)
        {
        case CATEGORY:

            found = text.find("#############################");
            if (found != std::string::npos)
            {
                STATE=NONE;
                continue;
            }
            else
            {
                processCategory(text);
                currentCategory=text;
            }

            break;
        case NONE:

            found = text.find("Uncomment");
            if (found != std::string::npos)
            {
                STATE=NONE;
                processDescription(text, found);
                description=text;
                continue;
            }
            found = text.find("DEFINES +=");
            if (found != std::string::npos)
            {
                STATE=NONE;
                bool activated=false;
                processOption(text, activated, found);
                DEFINES op(activated, text, description, currentCategory, true);
                std::vector< DEFINES >::iterator it = std::find(listOptions.begin(), listOptions.end(), op);
                if (it != listOptions.end())
                {
                    it->description=description;
                    it->category=currentCategory;
                    it->value=activated;
                }
                else
                {
                    listOptions.push_back(op);
                }


                listOptions.back().addConditions(conditions);
                continue;
            }

            //FIND {
            found = text.find("{");
            if (found != std::string::npos)
            {

                TYPE_CONDITION type;
                bool presence;
                processCondition(text, presence,type,found);
                conditions.push_back(CONDITION(type,presence,text));
                STATE=NONE;
                continue;
            }
            found = text.find("#############################");
            if (found != std::string::npos)
            {

                STATE=CATEGORY;
                continue;
            }
            if (text[0]=='}')
            {
                conditions.pop_back();
                STATE=NONE;
                continue;
            }
            found = text.find('=');
            if (found != std::string::npos           &&
                text.find("<=") == std::string::npos &&
                text.find(">=") == std::string::npos    )
            {
                std::string name;
                bool presence;
                processTextOption(text, name, presence, found);
                DEFINES op(presence,name,text,currentCategory,false);
                std::vector< DEFINES >::iterator it = std::find(listOptions.begin(), listOptions.end(), op);
                if (it != listOptions.end())
                {
                    it->description=text;
                    it->category=currentCategory;
                    it->value=presence;
                }
                else
                {
                    listOptions.push_back(op);
                }



                listOptions.back().addConditions(conditions);
            }
            else
            {
                removeInitialCharacter(text,'#');
                removeInitialCharacter(text,' ');
                description+="\n"+text;
            }
            continue;

            std::cerr << "NOT FOUND: " << text << "\n";

        }

    }
}

//Copy/Paste of the content of helper/system/SetDirectory.cpp
// Get the full path of the current process. The given filename should be the value of argv[0].
std::string GetProcessFullPath(const char* filename)
{
#if defined (WIN32)
    if (!filename || !filename[0])
    {
        //return __argv[0];
        int n=0;
        LPWSTR wpath = *CommandLineToArgvW(GetCommandLineW(),&n);
        if (wpath)
        {
            char path[1024];
            memset(path,0,sizeof(path));
            wcstombs(path, wpath, sizeof(path)-1);
            //std::cout << "Current process: "<<path<<std::endl;
            if (path[0]) return path;
        }
    }
    /// \TODO use GetCommandLineW and/or CommandLineToArgvW. This is however not strictly necessary, as argv[0] already contains the full path in most cases.
#elif defined (__linux__)
    if (!filename || filename[0]!='/')
    {
        char path[1024];
        memset(path,0,sizeof(path));
        readlink("/proc/self/exe",path,sizeof(path)-1);
// 		std::cout << "Current process: "<< path <<std::endl;
        if (path[0])
            return path;
        else
            std::cout << "ERROR: can't get current process path..." << std::endl;
    }
#elif defined (__APPLE__)
    if (!filename || filename[0]!='/')
    {
        char path[1024];
        CFBundleRef mainBundle = CFBundleGetMainBundle();
        assert(mainBundle);

        CFURLRef mainBundleURL = CFBundleCopyBundleURL(mainBundle);
        assert(mainBundleURL);

        CFStringRef cfStringRef = CFURLCopyFileSystemPath( mainBundleURL, kCFURLPOSIXPathStyle);
        assert(cfStringRef);

        CFStringGetCString(cfStringRef, path, 1024, kCFStringEncodingASCII);

        CFRelease(mainBundleURL);
        CFRelease(cfStringRef);

        return std::string(path);
    }
#endif

    return filename;
}

int main(int argc, char** argv)
{

    std::string file;
    file=GetProcessFullPath(argv[0]);

    std::size_t bin = file.find("bin");

    if (bin != std::string::npos)
    {
        file.resize(bin-1);
    }
    else
    {

        std::cerr << "ERROR: $SOFA/bin directory not FOUND!" << std::endl;
        return 1;
    }

    std::cerr << "Using " <<file << " as path for Sofa" << std::endl;

    std::ifstream sofa_default((file+"/sofa-default.cfg").c_str());
    std::ifstream sofa_local((file+"/sofa-local.cfg").c_str());

    typedef std::vector<DEFINES> VecDEFINES;
    VecDEFINES  listOptions;

    parse(sofa_default, listOptions);

    if (sofa_local.good())
    {
        //Set to false all the option
        for (unsigned int i=0; i<listOptions.size(); ++i) listOptions[i].value=false;
        parse(sofa_local, listOptions);
    }

    sofa_default.close();
    sofa_local.close();

    QApplication* application;
    application = new QApplication(argc, argv);

    sofa::gui::qt::SofaConfiguration* config = new sofa::gui::qt::SofaConfiguration(file,listOptions);
    application->setMainWidget(config);

    config->show();
    return application->exec();
}
