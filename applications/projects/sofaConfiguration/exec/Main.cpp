
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

#include "../lib/SofaConfiguration.h"
#include "../lib/ConfigurationParser.h"


#include <qapplication.h>
#include <qpixmap.h>

#include <iostream>
#include <fstream>

using sofa::gui::qt::DEFINES;

//Copy/Paste of the content of helper/system/SetDirectory.cpp
// Get the full path of the current process. The given filename should be the value of argv[0].
std::string GetProcessFullPath(const char* filename)
{
#if defined (WIN32)
    if (!filename || !filename[0])
    {
        TCHAR tpath[1024];
        GetModuleFileName(NULL,tpath,1024);
        std::wstring wprocessPath = tpath;
        std::string processPath;
        processPath.assign(wprocessPath.begin(), wprocessPath.end() );
        std::cout << "Current process: "<<processPath<<std::endl;
        return processPath;
    }
    /// \TODO use GetCommandLineW and/or CommandLineToArgvW. This is however not strictly necessary, as argv[0] already contains the full path in most cases.
#elif defined (__linux__)
    if (!filename || filename[0]!='/')
    {
        char path[1024];
        memset(path,0,sizeof(path));
        ssize_t l=readlink("/proc/self/exe",path,sizeof(path)-1);
// 		std::cout << "Current process: "<< path <<std::endl;
        if (l != -1 && path[0])
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
    file=GetProcessFullPath("");

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

    // std::cerr << "Using " <<file << " as path for Sofa" << std::endl;

    std::ifstream sofa_default((file+"/sofa-default.cfg").c_str());
    std::ifstream sofa_local((file+"/sofa-local.cfg").c_str());

    typedef std::vector<DEFINES> VecDEFINES;
    VecDEFINES  listOptions;

    sofa::gui::qt::ConfigurationParser::Parse(sofa_default, listOptions);

    if (sofa_local.good())
    {
        for (unsigned int i=0; i<listOptions.size(); ++i) listOptions[i].value=false;
        sofa::gui::qt::ConfigurationParser::Parse(sofa_local, listOptions);
    }

    sofa_default.close();
    sofa_local.close();

    QApplication* application;
    application = new QApplication(argc, argv);

    sofa::gui::qt::SofaConfiguration* config = new sofa::gui::qt::SofaConfiguration(file,listOptions);
    application->setMainWidget(config);

    config->show();

    //Setting the icon
    QString pathIcon=(file + std::string( "/share/icons/SOFACONFIGURATION.png" )).c_str();
#ifdef SOFA_QT4
    application->setWindowIcon(QIcon(pathIcon));
#else
    config->setIcon(QPixmap(pathIcon));
#endif

    return application->exec();
}
