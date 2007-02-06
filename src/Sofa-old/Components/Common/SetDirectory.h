#ifndef SOFA_COMPONENTS_COMMON_SETDIRECTORY_H
#define SOFA_COMPONENTS_COMMON_SETDIRECTORY_H

namespace Sofa
{

namespace Components
{

namespace Common
{

// A small utility class to temporarly set the current directory to the same as a specified file
class SetDirectory
{
public:
    char previousDir[1024];
    char* directory;

    SetDirectory(const char* filename);

    ~SetDirectory();
};

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
