#ifndef SOFA_COMPONENTS_GL_CAPTURE_H
#define SOFA_COMPONENTS_GL_CAPTURE_H

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <GL/gl.h>

#include "../Common/Image.h"

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

class Capture
{
protected:
    std::string prefix;
    int counter;

public:

    Capture();

    const std::string& getPrefix() const { return prefix; }
    int getCounter() const { return counter; }

    void setPrefix(const std::string v) { prefix=v; }
    void setCounter(int v=-1) { counter = v; }

    bool saveScreen(const std::string& filename);

    bool saveScreen();
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
