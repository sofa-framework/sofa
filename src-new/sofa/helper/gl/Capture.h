#ifndef SOFA_HELPER_GL_CAPTURE_H
#define SOFA_HELPER_GL_CAPTURE_H

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <GL/gl.h>

#include <sofa/helper/io/Image.h>

namespace sofa
{

namespace helper
{

namespace gl
{

//using namespace sofa::defaulttype;

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

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
