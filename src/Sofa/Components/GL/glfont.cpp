#ifdef WIN32
# include <windows.h>
#else
# include <X11/Xlib.h>
# include <X11/Xutil.h>
# include <GL/glx.h>
#endif

#include <string.h>
#include <GL/gl.h>

#include "glfont.h"

#ifdef SGI_CC
extern "C" Display * glXGetCurrentDisplayEXT (void);
#endif

namespace Sofa
{

namespace Components
{

namespace GL
{

static GLuint LettersDL=0;

#ifdef WIN32

void glfntInit(void)
{
    HDC hdc;

    if (LettersDL!=0) return;
    hdc=wglGetCurrentDC();
    SelectObject(hdc,GetStockObject(SYSTEM_FONT));
    LettersDL=glGenLists(128);
    wglUseFontBitmaps(hdc,32,127,LettersDL+32);
}

void glfntClose(void)
{
    if (LettersDL==0) return;
    glDeleteLists(LettersDL,128);
    LettersDL=0;
}

#else

static unsigned int last;

void glfntInit(void)
{

    Display *dpy;

    XFontStruct *fontInfo;
    Font id;
    unsigned int first;

    if (LettersDL!=0) return;
#ifdef SGI_CC
    dpy=glXGetCurrentDisplayEXT();
#else
    dpy=glXGetCurrentDisplay();
#endif
    if (dpy==NULL) return;
    fontInfo = XLoadQueryFont(dpy,
            "-adobe-times-medium-r-normal--17-120-100-100-p-88-iso8859-1");
    if (fontInfo == NULL) return;

    id = fontInfo->fid;
    first = fontInfo->min_char_or_byte2;
    last = fontInfo->max_char_or_byte2;
    if (first<32) first=32;
    if (last>127) last=127;

    LettersDL=glGenLists(last+1);
    if (LettersDL==0) return;
    glXUseXFont(id, first, last-first+1, LettersDL+first);

}

void glfntClose(void)
{
    if (LettersDL==0) return;
    glDeleteLists(LettersDL,last+1);
    LettersDL=0;
}


#endif

void glfntWriteBitmap(float x,float y,char *s)
{
    GLint CurBase;

    if (LettersDL==0) return;
    glRasterPos2f(x,y);

    glGetIntegerv(GL_LIST_BASE,&CurBase);
    glListBase(LettersDL);
    glCallLists(strlen(s),GL_UNSIGNED_BYTE,s);
    glListBase(CurBase);
}

} // namespace GL

} // namespace Components

} // namespace Sofa
