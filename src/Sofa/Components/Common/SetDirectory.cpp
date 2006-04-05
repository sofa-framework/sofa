#include "SetDirectory.h"

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <direct.h>
#endif

#include <string.h>


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
#ifndef WIN32
        getcwd(previousDir,sizeof(previousDir));
        chdir(directory);
#else
        _getcwd(previousDir, 1024);
        _chdir(directory);
#endif
    }
}

SetDirectory::~SetDirectory()
{
    if (directory[0] && previousDir[0])
    {
#ifndef WIN32
        chdir(previousDir);
#else
        _chdir(directory);
#endif
    }
    delete[] directory;
}

} // namespace Common

} // namespace Components

} // namespace Sofa
