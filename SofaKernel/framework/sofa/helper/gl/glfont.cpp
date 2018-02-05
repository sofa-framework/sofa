/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/gl.h>
#if defined (__linux__)
# include <X11/Xlib.h>
# include <X11/Xutil.h>
# include <GL/glx.h>
#endif

#include <string.h>


#include <sofa/helper/gl/glfont.h>

#ifdef SGI_CC
extern "C" Display * glXGetCurrentDisplayEXT (void);
#endif

namespace sofa
{

namespace helper
{

namespace gl
{

#if __APPLE__

// nothing yet

void glfntInit(void) {}
void glfntClose(void) {}
void glfntWriteBitmap(float /*x*/, float /*y*/, char * /*s*/ ) {}

#endif

#ifdef WIN32

static GLuint LettersDL=0;

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

void glfntWriteBitmap(float x,float y,char *s)
{
    GLint CurBase;

    if (LettersDL==0) return;
    glRasterPos2f(x,y);

    glGetIntegerv(GL_LIST_BASE,&CurBase);
    glListBase(LettersDL);
    glCallLists((GLsizei)strlen(s),GL_UNSIGNED_BYTE,s);
    glListBase(CurBase);
}
#endif // WIN32

#ifdef __linux__

static GLuint LettersDL=0;

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

#endif // __linux__



} // namespace gl

} // namespace helper

} // namespace sofa

