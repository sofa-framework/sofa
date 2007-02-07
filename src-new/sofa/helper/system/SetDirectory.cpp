#include "SetDirectory.h"

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <direct.h>
#endif

#include <string.h>
#include <iostream>


namespace Sofa
{

namespace Components
{

namespace Common
{

SetDirectory::SetDirectory(const char* filename)
{
    int len = strlen(filename);
    while (len>0 && filename[len]!='\\' && filename[len]!='/')
        --len;
    directory = new char[len+1];
    memcpy(directory, filename, len);
    directory[len]='\0';
    previousDir[0]='\0';
    if (directory[0])
    {
        std::cout << "chdir("<<directory<<")"<<std::endl;
#ifndef WIN32
        getcwd(previousDir, sizeof(previousDir));
        chdir(directory);
#else
        _getcwd(previousDir, sizeof(previousDir));
        _chdir(directory);
#endif
    }
}

SetDirectory::~SetDirectory()
{
    if (directory[0] && previousDir[0])
    {
        std::cout << "chdir("<<previousDir<<")"<<std::endl;
#ifndef WIN32
        chdir(previousDir);
#else
        _chdir(previousDir);
#endif
    }
    delete[] directory;
}

} // namespace Common

} // namespace Components

} // namespace Sofa
