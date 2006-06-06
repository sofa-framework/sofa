#include "Capture.h"
#include "../ImageBMP.h"

#include <sys/types.h>
#include <sys/stat.h>

namespace Sofa
{

namespace Components
{

namespace GL
{

Capture::Capture()
    : prefix("capture"), counter(-1)
{
}

bool Capture::saveScreen(const std::string& filename)
{
    ImageBMP img;
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    img.init(viewport[2],viewport[3],24);
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, img.getData());
    if (!img.save(filename)) return false;
    std::cout << "Saved "<<img.getWidth()<<"x"<<img.getHeight()<<" screen image to "<<filename<<std::endl;
    return true;
}

bool Capture::saveScreen()
{
    std::string filename;
    char buf[32];
    int c;
    if (counter == -1)
    {
        c = 0;
        struct stat st;
        do
        {
            ++c;
            sprintf(buf, "%04d",c);
            filename = prefix;
            filename += buf;
            filename += ".bmp";
        }
        while (stat(filename.c_str(),&st)==0);
    }
    else
    {
        c = counter++;
    }
    sprintf(buf, "%04d",c);
    filename = prefix;
    filename += buf;
    filename += ".bmp";
    return saveScreen(filename);
}

} // namespace GL

} // namespace Components

} // namespace Sofa
