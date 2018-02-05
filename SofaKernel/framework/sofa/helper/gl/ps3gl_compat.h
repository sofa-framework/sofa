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

#ifndef _PS3GL_COMPAT_H_
#define _PS3GL_COMPAT_H_

#include <PSGL\psgl.h>
#include <PSGL\psglu.h>
#include <GLES\glext.h>
#include <math.h>
#include <vector>

// Used as key in sofa in global map of shaders
#define GL_VERTEX_SHADER_ARB	0x8B31
#define GL_FRAGMENT_SHADER_ARB	0x8B30

// Dummy types and functions for compilation of shaders (de-activated on PS3)
typedef uint32_t GLhandleARB;
typedef char GLcharARB;

#define GL_PROGRAM_OBJECT_ARB 0xffff
#define GL_OBJECT_COMPILE_STATUS_ARB 0xffff
#define GL_OBJECT_INFO_LOG_LENGTH_ARB 0xffff
#define GL_VERTEX_PROGRAM_TWO_SIDE 0xffff
#define GL_CLAMP_VERTEX_COLOR 0xffff
#define GL_OBJECT_LINK_STATUS_ARB 0xffff
#define GL_ENABLE_BIT 0xffff
#define GL_LIGHTING_BIT 0xffff
#define GL_ALL_ATTRIB_BITS 0xffff
#define GL_LIGHT_MODEL_LOCAL_VIEWER 0xffff
#define GL_POLYGON_SMOOTH 0x0B41
#define GL_RENDER 1
#define gluOrtho2D gluOrtho2Df

inline void glUseProgramObjectARB(GLhandleARB handle){}
inline void  glShaderSourceARB(GLhandleARB, int , const char** , void* ){}
inline void glCompileShaderARB(GLhandleARB){}
inline GLint getGeometryInputType(unsigned int) { return 0; }
inline void setGeometryInputType(unsigned int, GLint){}
inline void setGeometryVerticesOut(unsigned int, GLint){}
inline GLint GetGeometryVerticesOut() { return 0; }
inline GLint glGetHandleARB(GLint) { return 0; }
inline void glClampColorARB(GLuint, GLboolean) {}
inline GLhandleARB glCreateShaderObjectARB(GLint) { return 0; }
inline void glGetInfoLogARB(GLhandleARB object, GLsizei maxLength, GLsizei *length, GLcharARB *infoLog){}
inline void glDeleteObjectARB(GLhandleARB object) {}
inline GLhandleARB glCreateProgramObjectARB(void) { return 0; }
inline void glAttachObjectARB(GLhandleARB program, GLhandleARB shader) {}
inline void glLinkProgramARB(GLhandleARB program){}
inline void glGetObjectParameterivARB(GLhandleARB object, GLenum type, int *param) {}

inline void glUniform1fARB(GLint variable, float newValue) {}
inline void glUniform2fARB(GLint variable, float v0, float v1) {}
inline void glUniform3fARB(GLint variable, float v0, float v1, float v2) {}
inline void glUniform4fARB(GLint variable, float v0, float v1, float v2, float v3) {}
inline void glUniform1ivARB(GLint variable, GLsizei count, const GLint *value) {}
inline void glUniform2ivARB(GLint variable, GLsizei count, const GLint *value) {}
inline void glUniform3ivARB(GLint variable, GLsizei count, const GLint *value) {}
inline void glUniform4ivARB(GLint variable, GLsizei count, const GLint *value) {}
inline void glUniform1iARB(GLint variable, int newValue) {}
inline void glUniform2iARB(GLint variable, int i1, int i2) {}
inline void glUniform3iARB(GLint variable, int i1, int i2, int i3) {}
inline void glUniform4iARB(GLint variable, int i1, int i2, int i3, int i4) {}
inline void glUniform1fvARB(GLint variable, GLsizei count, const float *value) {}
inline void glUniform2fvARB(GLint variable, GLsizei count, const float *value) {}
inline void glUniform3fvARB(GLint variable, GLsizei count, const float *value) {}
inline void glUniform4fvARB(GLint variable, GLsizei count, const float *value) {}
inline void glUniformMatrix2fv(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix3fv(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix4fv(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix2x3fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix3x2fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix2x4fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix4x2fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix3x4fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline void glUniformMatrix4x3fv(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) {}
inline GLint glGetUniformLocationARB(GLhandleARB program, const GLcharARB *name) { return 0; }
inline GLint glGetAttribLocationARB(GLhandleARB program,const GLcharARB *name) { return 0; }
inline void glDetachObjectARB(GLhandleARB program, GLhandleARB shader) {}
inline void glPushAttrib(GLuint attr) {}
inline void glPopAttrib() {}
inline void glCallList(GLuint list) {}

#define GLUT_STROKE_ROMAN 0xffff
inline void glutStrokeCharacter(GLuint , char) {}
inline GLuint glutStrokeWidth(GLuint, char) { return 0; }

// Double not supported
inline void glTranslated(double x, double y, double z)
{
	glTranslatef((float)x, (float) y, (float) z);
}

// Display lists
inline GLuint glGenLists(GLsizei range)
{
	return 0;
}

inline void glNewList(GLuint list,GLenum mode)
{
}

inline void glEndList(void)
{
}

#define GL_COMPILE 0xffff

// Opengl ES remappings
#define glGenFramebuffersEXT glGenFramebuffersOES
#define glBindFramebufferEXT glBindFramebufferOES
#define glBindRenderbuffersEXT glBindRenderbuffersOES
#define glDeleteFramebuffersEXT glDeleteFramebuffersOES
#define glGenRenderbuffersEXT glGenRenderbuffersOES
#define glDeleteRenderbuffersEXT glDeleteRenderbuffersOES
#define glCheckFramebufferStatusEXT glCheckFramebufferStatusOES
#define glBindRenderbufferEXT glBindRenderbufferOES
#define glRenderbufferStorageEXT glRenderbufferStorageOES
#define GL_FRAMEBUFFER_EXT GL_FRAMEBUFFER_OES
#define GL_FRAMEBUFFER_COMPLETE_EXT GL_FRAMEBUFFER_COMPLETE_OES
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_OES
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_OES
#define GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_OES
#define GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT GL_FRAMEBUFFER_INCOMPLETE_FORMATS_OES
#define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT 0xfff2
#define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT 0xfff1
#define GL_FRAMEBUFFER_UNSUPPORTED_EXT GL_FRAMEBUFFER_UNSUPPORTED_OES
#define glFramebufferTexture2DEXT glFramebufferTexture2DOES
#define glFramebufferRenderbufferEXT glFramebufferRenderbufferOES
#define GL_DEPTH_ATTACHMENT_EXT GL_FRAMEBUFFER_OES
#define GL_RENDERBUFFER_EXT GL_RENDERBUFFER_OES
#define GL_DEPTH_TEXTURE_MODE GL_DEPTH_TEXTURE_MODE_ARB
#define glVertex3d glVertex3f
#define glVertex2d glVertex2f
#define glNormal3d glNormal3f
#define glTexCoord3d glTexCoord3f
#define glTexCoord2d glTexCoord2f
#define glScaled glScalef
#define glRotated glRotatef
#define GLdouble GLfloat
#define GL_VIEWPORT GL_MAX_VIEWPORT_DIMS
#define GL_TEXTURE_1D GL_TEXTURE_2D
#define glOrtho glOrthof
#define glBindBufferARB glBindBuffer
#define glGenBuffersARB glGenBuffers
#define glBufferSubDataARB glBufferSubData
#define gluLookAt gluLookAtf
#define gluPerspective gluPerspectivef
#define glTranslate glTranslatef
#define glClearDepth glClearDepthf
#define glLoadMatrixd glLoadMatrixf
#define glDeleteBuffersARB glDeleteBuffers
#define glBufferDataARB glBufferData
#define glLightModeli glLightModelf

/*	In order to emulate immediate draw on PS3 using glBegin and glEnd :
	we create a user command buffer and write directly the primitive begin end command inside it since
	the psgl API is lacking it !
	The minimal amount to allocate is 1MB but here we allocate more because the command buffer will store the verts...
*/

//---------------------------------------------------------------------------------------------------------------------
// GL Missing functions
//---------------------------------------------------------------------------------------------------------------------

// GLUT
	/* Display mode bit masks. */ 
	#define GLUT_RGB			0
	#define GLUT_RGBA			GLUT_RGB
	#define GLUT_INDEX			1
	#define GLUT_SINGLE			0
	#define GLUT_DOUBLE			2
	#define GLUT_ACCUM			4
	#define GLUT_ALPHA			8
	#define GLUT_DEPTH			16
	#define GLUT_STENCIL			32
	#if (GLUT_API_VERSION >= 2)
	#define GLUT_MULTISAMPLE		128
	#define GLUT_STEREO			256
	#endif				/*  */
	#if (GLUT_API_VERSION >= 3)
	#define GLUT_LUMINANCE			512
	#endif				/*  */

	/* Mouse buttons. */ 
	#define GLUT_LEFT_BUTTON		0
	#define GLUT_MIDDLE_BUTTON		1
	#define GLUT_RIGHT_BUTTON		2

	/* Mouse button  state. */ 
	#define GLUT_DOWN			0
	#define GLUT_UP				1

	#if (GLUT_API_VERSION >= 2)
	/* function keys */ 
	#define GLUT_KEY_F1			1
	#define GLUT_KEY_F2			2
	#define GLUT_KEY_F3			3
	#define GLUT_KEY_F4			4
	#define GLUT_KEY_F5			5
	#define GLUT_KEY_F6			6
	#define GLUT_KEY_F7			7
	#define GLUT_KEY_F8			8
	#define GLUT_KEY_F9			9
	#define GLUT_KEY_F10			10
	#define GLUT_KEY_F11			11
	#define GLUT_KEY_F12			12
	/* directional keys */ 
	#define GLUT_KEY_LEFT			100
	#define GLUT_KEY_UP			101
	#define GLUT_KEY_RIGHT			102
	#define GLUT_KEY_DOWN			103
	#define GLUT_KEY_PAGE_UP		104
	#define GLUT_KEY_PAGE_DOWN		105
	#define GLUT_KEY_HOME			106
	#define GLUT_KEY_END			107
	#define GLUT_KEY_INSERT			108
	#endif				/*  */

	/* Entry/exit  state. */ 
	#define GLUT_LEFT			0
	#define GLUT_ENTERED			1

	/* Menu usage  state. */ 
	#define GLUT_MENU_NOT_IN_USE		0
	#define GLUT_MENU_IN_USE		1

	/* Visibility  state. */ 
	#define GLUT_NOT_VISIBLE		0
	#define GLUT_VISIBLE			1

	/* Window status  state. */ 
	#define GLUT_HIDDEN			0
	#define GLUT_FULLY_RETAINED		1
	#define GLUT_PARTIALLY_RETAINED		2
	#define GLUT_FULLY_COVERED		3

	/* Color index component selection values. */ 
	#define GLUT_RED			0
	#define GLUT_GREEN			1
	#define GLUT_BLUE			2

	/* Layers for use. */ 
	#define GLUT_NORMAL			0
	#define GLUT_OVERLAY			1

		/* glutGet parameters. */ 
	#define GLUT_WINDOW_X			100
	#define GLUT_WINDOW_Y			101
	#define GLUT_WINDOW_WIDTH		102
	#define GLUT_WINDOW_HEIGHT		103
	#define GLUT_WINDOW_BUFFER_SIZE		104
	#define GLUT_WINDOW_STENCIL_SIZE	105
	#define GLUT_WINDOW_DEPTH_SIZE		106
	#define GLUT_WINDOW_RED_SIZE		107
	#define GLUT_WINDOW_GREEN_SIZE		108
	#define GLUT_WINDOW_BLUE_SIZE		109
	#define GLUT_WINDOW_ALPHA_SIZE		110
	#define GLUT_WINDOW_ACCUM_RED_SIZE	111
	#define GLUT_WINDOW_ACCUM_GREEN_SIZE	112
	#define GLUT_WINDOW_ACCUM_BLUE_SIZE	113
	#define GLUT_WINDOW_ACCUM_ALPHA_SIZE	114
	#define GLUT_WINDOW_DOUBLEBUFFER	115
	#define GLUT_WINDOW_RGBA		116
	#define GLUT_WINDOW_PARENT		117
	#define GLUT_WINDOW_NUM_CHILDREN	118
	#define GLUT_WINDOW_COLORMAP_SIZE	119
	#if (GLUT_API_VERSION >= 2)
	#define GLUT_WINDOW_NUM_SAMPLES		120
	#define GLUT_WINDOW_STEREO		121
	#endif				/*  */
	#if (GLUT_API_VERSION >= 3)
	#define GLUT_WINDOW_CURSOR		122
	#endif				/*  */
	#define GLUT_SCREEN_WIDTH		200
	#define GLUT_SCREEN_HEIGHT		201
	#define GLUT_SCREEN_WIDTH_MM		202
	#define GLUT_SCREEN_HEIGHT_MM		203
	#define GLUT_MENU_NUM_ITEMS		300
	#define GLUT_DISPLAY_MODE_POSSIBLE	400
	#define GLUT_INIT_WINDOW_X		500
	#define GLUT_INIT_WINDOW_Y		501
	#define GLUT_INIT_WINDOW_WIDTH		502
	#define GLUT_INIT_WINDOW_HEIGHT		503
	#define GLUT_INIT_DISPLAY_MODE		504
	#if (GLUT_API_VERSION >= 2)
	#define GLUT_ELAPSED_TIME		700
	#endif				/*  */
	#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)
	#define GLUT_WINDOW_FORMAT_ID		123
	#endif				/*  */

	#if (GLUT_API_VERSION >= 2)
	/* glutDeviceGet parameters. */ 
	#define GLUT_HAS_KEYBOARD		600
	#define GLUT_HAS_MOUSE			601
	#define GLUT_HAS_SPACEBALL		602
	#define GLUT_HAS_DIAL_AND_BUTTON_BOX	603
	#define GLUT_HAS_TABLET			604
	#define GLUT_NUM_MOUSE_BUTTONS		605
	#define GLUT_NUM_SPACEBALL_BUTTONS	606
	#define GLUT_NUM_BUTTON_BOX_BUTTONS	607
	#define GLUT_NUM_DIALS			608
	#define GLUT_NUM_TABLET_BUTTONS		609
	#endif				/*  */
	#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)
	#define GLUT_DEVICE_IGNORE_KEY_REPEAT   610
	#define GLUT_DEVICE_KEY_REPEAT  		611
	#define GLUT_HAS_JOYSTICK		612
	#define GLUT_OWNS_JOYSTICK		613
	#define GLUT_JOYSTICK_BUTTONS		614
	#define GLUT_JOYSTICK_AXES		615
	#define GLUT_JOYSTICK_POLL_RATE		616
	#endif				/*  */

	#if (GLUT_API_VERSION >= 3)
	/* glutLayerGet parameters. */ 
	#define GLUT_OVERLAY_POSSIBLE   		800
	#define GLUT_LAYER_IN_USE		801
	#define GLUT_HAS_OVERLAY		802
	#define GLUT_TRANSPARENT_INDEX		803
	#define GLUT_NORMAL_DAMAGED		804
	#define GLUT_OVERLAY_DAMAGED		805

	#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
	/* glutVideoResizeGet parameters. */ 
	#define GLUT_VIDEO_RESIZE_POSSIBLE	900
	#define GLUT_VIDEO_RESIZE_IN_USE	901
	#define GLUT_VIDEO_RESIZE_X_DELTA	902
	#define GLUT_VIDEO_RESIZE_Y_DELTA	903
	#define GLUT_VIDEO_RESIZE_WIDTH_DELTA	904
	#define GLUT_VIDEO_RESIZE_HEIGHT_DELTA	905
	#define GLUT_VIDEO_RESIZE_X		906
	#define GLUT_VIDEO_RESIZE_Y		907
	#define GLUT_VIDEO_RESIZE_WIDTH		908
	#define GLUT_VIDEO_RESIZE_HEIGHT	909
	#endif				/*  */

	/* glutUseLayer parameters. */ 
	#define GLUT_NORMAL			0
	#define GLUT_OVERLAY			1

	/* glutGetModifiers return mask. */ 
	#define GLUT_ACTIVE_SHIFT   			1
	#define GLUT_ACTIVE_CTRL				2
	#define GLUT_ACTIVE_ALT 				4

	/* glutSetCursor parameters. */ 
	/* Basic arrows. */ 
	#define GLUT_CURSOR_RIGHT_ARROW		0
	#define GLUT_CURSOR_LEFT_ARROW		1
	/* Symbolic cursor shapes. */ 
	#define GLUT_CURSOR_INFO		2
	#define GLUT_CURSOR_DESTROY		3
	#define GLUT_CURSOR_HELP		4
	#define GLUT_CURSOR_CYCLE		5
	#define GLUT_CURSOR_SPRAY		6
	#define GLUT_CURSOR_WAIT		7
	#define GLUT_CURSOR_TEXT		8
	#define GLUT_CURSOR_CROSSHAIR		9
	/* Directional cursors. */ 
	#define GLUT_CURSOR_UP_DOWN		10
	#define GLUT_CURSOR_LEFT_RIGHT		11
	/* Sizing cursors. */ 
	#define GLUT_CURSOR_TOP_SIDE		12
	#define GLUT_CURSOR_BOTTOM_SIDE		13
	#define GLUT_CURSOR_LEFT_SIDE		14
	#define GLUT_CURSOR_RIGHT_SIDE		15
	#define GLUT_CURSOR_TOP_LEFT_CORNER	16
	#define GLUT_CURSOR_TOP_RIGHT_CORNER	17
	#define GLUT_CURSOR_BOTTOM_RIGHT_CORNER	18
	#define GLUT_CURSOR_BOTTOM_LEFT_CORNER	19
	/* Inherit from parent window. */ 
	#define GLUT_CURSOR_INHERIT		100
	/* Blank cursor. */ 
	#define GLUT_CURSOR_NONE		101
	/* Fullscreen crosshair (if available). */ 
	#define GLUT_CURSOR_FULL_CROSSHAIR	102
	#endif				/*  */

extern void (*glutDisplay)();
extern void (*glutIdle)();


inline void glutSwapBuffers()
{
	psglSwap();
}

inline void glutReshapeFunc ( void(*glut_reshape)(void) )
{
}
inline void glutIdleFunc ( void(*glut_idle)(void) )
{
	glutIdle = glut_idle;
}
inline void glutDisplayFunc ( void(*glut_display)(void) )
{
	glutDisplay = glut_display;
}
inline void glutKeyboardFunc ( void(*glut_keyboard)(void) )
{
}
inline void glutSpecialFunc ( void(*glut_special)(void) )
{
}
inline void glutMouseFunc ( void(*glut_mouse)(void) )
{
}
inline void glutMotionFunc ( void(*glut_motion)(void) )
{
}

inline void glutPassiveMotionFunc ( void(*glut_motion)(void) )
{
}



inline void glutSolidCone(float base, float height, int slices, int stacks)
{

}

inline void glutMainLoop()
{
	while(true)
	{
		glutIdle();
		glutDisplay();
	}
}

inline void glutPostRedisplay()
{
}

inline void glutWireCube(float size)
{
}


inline GLboolean glIsEnabled(GLuint flag)
{
	return false;
}

inline void glutCreateWindow(const char* pTitle)
{

}

inline void glutInit(int *argcp, char **argv)
{
}

inline void glutInitDisplayMode(GLuint flags)
{
	psglInit(NULL);
	PSGLdevice* dev;// = psglCreateDeviceAuto(GL_ARGB_SCE, GL_DEPTH_COMPONENT24, GL_MULTISAMPLING_NONE_SCE);
	
	PSGLdeviceParameters params;
	params.enable =  PSGL_DEVICE_PARAMETERS_COLOR_FORMAT|PSGL_DEVICE_PARAMETERS_DEPTH_FORMAT|PSGL_DEVICE_PARAMETERS_WIDTH_HEIGHT|PSGL_DEVICE_PARAMETERS_BUFFERING_MODE;
	params.colorFormat = GL_ARGB_SCE;
	params.depthFormat = GL_DEPTH_COMPONENT24;
	params.width = 1280;
	params.height = 720;
	params.bufferingMode = PSGL_BUFFERING_MODE_DOUBLE;
	dev = psglCreateDeviceExtended(&params);

	GLuint screen_width;
	GLuint screen_height;
	psglGetDeviceDimensions(dev, &screen_width, &screen_height);
	PSGLcontext *context = psglCreateContext(); 
    
	if (!context) { 
            fprintf(stderr, "Error creating PSGL context\n"); 
            exit(1); } 
    
	psglMakeCurrent(context, dev);
    psglResetCurrentContext(); 
}

inline void glutBitmapCharacter(void* font, char c)
{

}

inline void glutSetWindowTitle(const char* pStr)
{
}


inline void glRasterPos2i(int x, int y)
{
}

inline void glTexImage1D(
  GLenum target,
  GLint level,
  GLint internalformat,
  GLsizei width,
  GLint border,
  GLint format,
  GLenum type,
  const GLvoid *pixels
)
{
	glTexImage2D(target, level, internalformat, width, 1, border, format, type, pixels);
}

inline void glDrawBuffer(GLuint mode) 
{
	if(mode==GL_BACK) 
	{
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	}
}

//---------------------------------------------------------------------------------------------------------------------
// glGetDoublev
//---------------------------------------------------------------------------------------------------------------------
inline void glGetDoublev(GLenum state, GLdouble* val)
{
	assert(state==GL_MODELVIEW_MATRIX || state==GL_PROJECTION_MATRIX && "Other types than matrix not supported please provide implementation");
	
	glGetFloatv(state, val);
}

//---------------------------------------------------------------------------------------------------------------------
// glGetDoublev
//---------------------------------------------------------------------------------------------------------------------
inline void glGetDoublev(GLenum state, double* val)
{
	assert(state==GL_MODELVIEW_MATRIX || state==GL_PROJECTION_MATRIX && "Other types than matrix not supported please provide implementation");
	float fMat[16];
	glGetFloatv(state, fMat);

	for(int i=0; i<16; i++)
	{
		fMat[i] = (float) val[i];
	}
}

//---------------------------------------------------------------------------------------------------------------------
// glMultMatrixd
//---------------------------------------------------------------------------------------------------------------------
inline void glMultMatrixd(double* mat)
{
	float f[16];
	
	for(int i=0; i<16; i++)
	{
		f[i] = (float) mat[i];
	}
	glMultMatrixf(f);
}

//---------------------------------------------------------------------------------------------------------------------
// glMultMatrixd
//---------------------------------------------------------------------------------------------------------------------
inline void glMultMatrixd(float* mat)
{
	glMultMatrixf(mat);
}

//---------------------------------------------------------------------------------------------------------------------
// ImmediateModeState
//---------------------------------------------------------------------------------------------------------------------
struct ImmediateModeState
{
	ImmediateModeState();

	void Reset();
	void Flush();

	float* m_vertices;
	float* m_normals;
	float* m_texcoords;
	float* m_colors;

	unsigned char m_ComponentCountPerVertices;
	unsigned char m_ComponentCountPerColors;
	unsigned char m_ComponentCountPerTexCoords;

	GLuint m_mode;
	int m_numVertices;
	int m_numTexcoords;
	int m_numColors;
	int m_numNormals;
	bool m_bDefaultColor;
	bool m_IsInsideBegin;

};

extern ImmediateModeState glImmediateMode;

//---------------------------------------------------------------------------------------------------------------------
// glBegin
//---------------------------------------------------------------------------------------------------------------------
inline void glBegin(GLuint mode)
{
	glImmediateMode.Reset();
	glImmediateMode.m_mode = mode;
	glImmediateMode.m_IsInsideBegin = true;

}

inline void glEnd()
{
	glImmediateMode.Flush();
	glImmediateMode.m_IsInsideBegin = false;
}

//---------------------------------------------------------------------------------------------------------------------
// GL Missing functions
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
// glVertex3f
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex3f(float x, float y, float z)
{
	glImmediateMode.m_ComponentCountPerVertices = 3;
	glImmediateMode.m_vertices[glImmediateMode.m_numVertices++] = x;
	glImmediateMode.m_vertices[glImmediateMode.m_numVertices++] = y;
	glImmediateMode.m_vertices[glImmediateMode.m_numVertices++] = z;
}

//---------------------------------------------------------------------------------------------------------------------
// glVertex2f
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex2f(float x, float y)
{
	glImmediateMode.m_ComponentCountPerVertices = 2;
	glImmediateMode.m_vertices[glImmediateMode.m_numVertices++] = x;
	glImmediateMode.m_vertices[glImmediateMode.m_numVertices++] = y;	
}

//---------------------------------------------------------------------------------------------------------------------
// glVertex3i
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex3i(int x, int y , int z)
{
	glVertex3f(x, y, z);
}

//---------------------------------------------------------------------------------------------------------------------
// glTexCoord3f
//---------------------------------------------------------------------------------------------------------------------
inline void glTexCoord3f(float x, float y, float z)
{
	glImmediateMode.m_ComponentCountPerTexCoords = 3;

	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = x;
	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = y;
	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = z;
}

//---------------------------------------------------------------------------------------------------------------------
// glTexCoord2f
//---------------------------------------------------------------------------------------------------------------------
inline void glTexCoord2f(float x, float y)
{
	glImmediateMode.m_ComponentCountPerTexCoords = 2;

	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = x;
	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = y;
}
	
inline void glTexCoord1f(float x)
{
	glImmediateMode.m_ComponentCountPerTexCoords = 2;

	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = x;
	glImmediateMode.m_texcoords[glImmediateMode.m_numTexcoords++] = 0.0f;
}

//---------------------------------------------------------------------------------------------------------------------
// glVertex3fv
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex3fv(const GLvoid* f)
{
	glVertexPointer(3, GL_FLOAT, 0, (GLvoid*)f);
}

//---------------------------------------------------------------------------------------------------------------------
// glVertex3dv
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex3dv(const GLvoid* d)
{
	float f[3];
	for(int i=0; i< 3; i++)
	{
		f[i] = (float)(((double*)d)[i]);
	}

	glVertex3f(f[0], f[1], f[2]);
}

//---------------------------------------------------------------------------------------------------------------------
// glVertex2i
//---------------------------------------------------------------------------------------------------------------------
inline void glVertex2i(GLuint x, GLuint y) 
{ 
	glVertex3f((float)x, (float)y, 0); 
}

//---------------------------------------------------------------------------------------------------------------------
// glColor3f
//---------------------------------------------------------------------------------------------------------------------
inline void glColor3f(float r, float g, float b)
{
	glImmediateMode.m_bDefaultColor = !glImmediateMode.m_IsInsideBegin;

	glImmediateMode.m_ComponentCountPerColors = 3;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = r;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = g;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = b;
}

inline void glColor3d(double r, double g, double b)
{
	glImmediateMode.m_ComponentCountPerColors = 3;

	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = r;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = g;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = b;
}

//---------------------------------------------------------------------------------------------------------------------
// glColor4f
//---------------------------------------------------------------------------------------------------------------------
inline void glColor4f(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{
	glImmediateMode.m_ComponentCountPerColors = 4;

	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = r;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = g;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = b;
	glImmediateMode.m_colors[glImmediateMode.m_numColors++] = a;
}

//---------------------------------------------------------------------------------------------------------------------
// glColor4d
//---------------------------------------------------------------------------------------------------------------------
inline void glColor4d(double r, double g, double b, double a)
{
	glColor4f(r, g, b, a);
}

//---------------------------------------------------------------------------------------------------------------------
// glColor3fv
//---------------------------------------------------------------------------------------------------------------------
inline void glColor3fv(const GLvoid* d)
{
	float f[3];
	for(int i=0; i< 3; i++)
	{
		f[i] = (float)(((float*)d)[i]);
	}

	glColor3f(f[0], f[1], f[2]);
}


//---------------------------------------------------------------------------------------------------------------------
// glNormal3f
//---------------------------------------------------------------------------------------------------------------------
inline void glNormal3f(float x, float y, float z)
{
	glImmediateMode.m_normals[glImmediateMode.m_numNormals++] = x;
	glImmediateMode.m_normals[glImmediateMode.m_numNormals++] = y;
	glImmediateMode.m_normals[glImmediateMode.m_numNormals++] = z;
}

inline void glutSolidSphere(float radii, float stack, float slice)
{
//\todo PS3 stub to implement : glutSolidSphere;
}

inline void glColorMaterial(GLenum face, GLenum	mode)
{
	glEnable(GL_COLOR_MATERIAL);
}

inline void glMateriali(GLenum face, GLenum pname, int param)
{
	glMaterialf(GL_FRONT_AND_BACK, pname, (GLfloat)param);
}


static void __gluMultMatrixVecd(const GLdouble matrix[16], const GLdouble in[4],
		      GLdouble out[4])
{
    int i;

    for (i=0; i<4; i++) {
	out[i] = 
	    in[0] * matrix[0*4+i] +
	    in[1] * matrix[1*4+i] +
	    in[2] * matrix[2*4+i] +
	    in[3] * matrix[3*4+i];
    }
}
/*
** Invert 4x4 matrix.
** Contributed by David Moore (See Mesa bug #6748)
*/
static int __gluInvertMatrixd(const GLdouble m[16], GLdouble invOut[16])
{
    GLdouble inv[16], det;
    int i;

    inv[0] =   m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
             + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] =  -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
             - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] =   m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
             + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
             - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] =  -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
             - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] =   m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
             + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] =  -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
             - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
             + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] =   m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
             + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] =  -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
             - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
             + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
             - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] =  -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
             - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] =   m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
             + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
             - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
             + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0)
        return GL_FALSE;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return GL_TRUE;
}

static void __gluMultMatricesd(const GLdouble a[16], const GLdouble b[16],
				GLdouble r[16])
{
    int i, j;

    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    r[i*4+j] = 
		a[i*4+0]*b[0*4+j] +
		a[i*4+1]*b[1*4+j] +
		a[i*4+2]*b[2*4+j] +
		a[i*4+3]*b[3*4+j];
	}
    }
}


inline GLint gluUnProject(GLdouble winx, GLdouble winy, GLdouble winz,
		const GLdouble modelMatrix[16], 
		const GLdouble projMatrix[16],
                const GLint viewport[4],
	        GLdouble *objx, GLdouble *objy, GLdouble *objz)
{
    GLdouble finalMatrix[16];
    GLdouble in[4];
    GLdouble out[4];

    __gluMultMatricesd(modelMatrix, projMatrix, finalMatrix);
    if (!__gluInvertMatrixd(finalMatrix, finalMatrix)) return(GL_FALSE);

    in[0]=winx;
    in[1]=winy;
    in[2]=winz;
    in[3]=1.0;

    /* Map x and y from window coordinates */
    in[0] = (in[0] - viewport[0]) / viewport[2];
    in[1] = (in[1] - viewport[1]) / viewport[3];

    /* Map to range -1 to 1 */
    in[0] = in[0] * 2 - 1;
    in[1] = in[1] * 2 - 1;
    in[2] = in[2] * 2 - 1;

    __gluMultMatrixVecd(finalMatrix, in, out);
    if (out[3] == 0.0) return(GL_FALSE);
    out[0] /= out[3];
    out[1] /= out[3];
    out[2] /= out[3];
    *objx = out[0];
    *objy = out[1];
    *objz = out[2];
    return(GL_TRUE);
}

//---------------------------------------------------------------------------------------------------------------------
// GLU Quadrics
//---------------------------------------------------------------------------------------------------------------------

/* Make it not a power of two to avoid cache thrashing on the chip */
#define CACHE_SIZE	240

#undef	PI
#define PI	      3.14159265358979323846

/* QuadricDrawStyle */
#define GLU_POINT                          100010
#define GLU_LINE                           100011
#define GLU_FILL                           100012
#define GLU_SILHOUETTE                     100013

/* QuadricCallback */
/*      GLU_ERROR */

/* QuadricNormal */
#define GLU_SMOOTH                         100000
#define GLU_FLAT                           100001
#define GLU_NONE                           100002

/* QuadricOrientation */
#define GLU_OUTSIDE                        100020
#define GLU_INSIDE                         100021
#define GLU_ERROR                          100103

struct GLUquadric {
    GLint	normals;
    GLboolean	textureCoords;
    GLint	orientation;
    GLint	drawStyle;
    void	(*errorCallback)( GLint );
};

typedef GLUquadric GLUquadricObj;

/* Internal convenience typedefs */
typedef void (_GLUfuncptr)(void);

inline GLUquadric * gluNewQuadric(void)
{
    GLUquadric *newstate;

    newstate = (GLUquadric *) malloc(sizeof(GLUquadric));
    if (newstate == NULL) {
	/* Can't report an error at this point... */
	return NULL;
    }
    newstate->normals = GLU_SMOOTH;
    newstate->textureCoords = GL_FALSE;
    newstate->orientation = GLU_OUTSIDE;
    newstate->drawStyle = GLU_FILL;
    newstate->errorCallback = NULL;
    return newstate;
}


inline void gluDeleteQuadric(GLUquadric *state)
{
    free(state);
}

inline void gluQuadricError(GLUquadric *qobj, GLenum which)
{
    if (qobj->errorCallback) {
	qobj->errorCallback(which);
    }
}

inline void gluQuadricCallback(GLUquadric *qobj, GLenum which, _GLUfuncptr fn)
{
    switch (which) {
      case GLU_ERROR:
	qobj->errorCallback = (void (*)(GLint)) fn;
	break;
      default:
	gluQuadricError(qobj, GLU_INVALID_ENUM);
	return;
    }
}

inline void gluQuadricNormals(GLUquadric *qobj, GLenum normals)
{
    switch (normals) {
      case GLU_SMOOTH:
      case GLU_FLAT:
      case GLU_NONE:
	break;
      default:
	gluQuadricError(qobj, GLU_INVALID_ENUM);
	return;
    }
    qobj->normals = normals;
}

inline void gluQuadricTexture(GLUquadric *qobj, GLboolean textureCoords)
{
    qobj->textureCoords = textureCoords;
}

inline void gluQuadricOrientation(GLUquadric *qobj, GLenum orientation)
{
    switch(orientation) {
      case GLU_OUTSIDE:
      case GLU_INSIDE:
	break;
      default:
	gluQuadricError(qobj, GLU_INVALID_ENUM);
	return;
    }
    qobj->orientation = orientation;
}

inline void gluQuadricDrawStyle(GLUquadric *qobj, GLenum drawStyle)
{
    switch(drawStyle) {
      case GLU_POINT:
      case GLU_LINE:
      case GLU_FILL:
      case GLU_SILHOUETTE:
	break;
      default:
	gluQuadricError(qobj, GLU_INVALID_ENUM);
	return;
    }
    qobj->drawStyle = drawStyle;
}

inline void gluCylinder(GLUquadric *qobj, GLfloat baseRadius, GLfloat topRadius,
		GLfloat height, GLint slices, GLint stacks)
{
    GLint i,j;
    GLfloat sinCache[CACHE_SIZE];
    GLfloat cosCache[CACHE_SIZE];
    GLfloat sinCache2[CACHE_SIZE];
    GLfloat cosCache2[CACHE_SIZE];
    GLfloat sinCache3[CACHE_SIZE];
    GLfloat cosCache3[CACHE_SIZE];
    GLfloat angle;
    GLfloat zLow, zHigh;
    GLfloat sintemp, costemp;
    GLfloat length;
    GLfloat deltaRadius;
    GLfloat zNormal;
    GLfloat xyNormalRatio;
    GLfloat radiusLow, radiusHigh;
    int needCache2, needCache3;

    if (slices >= CACHE_SIZE) slices = CACHE_SIZE-1;

    if (slices < 2 || stacks < 1 || baseRadius < 0.0 || topRadius < 0.0 ||
	    height < 0.0) {
	gluQuadricError(qobj, GLU_INVALID_VALUE);
	return;
    }

    /* Compute length (needed for normal calculations) */
    deltaRadius = baseRadius - topRadius;
    length = sqrt(deltaRadius*deltaRadius + height*height);
    if (length == 0.0) {
	gluQuadricError(qobj, GLU_INVALID_VALUE);
	return;
    }

    /* Cache is the vertex locations cache */
    /* Cache2 is the various normals at the vertices themselves */
    /* Cache3 is the various normals for the faces */
    needCache2 = needCache3 = 0;
    if (qobj->normals == GLU_SMOOTH) {
	needCache2 = 1;
    }

    if (qobj->normals == GLU_FLAT) {
	if (qobj->drawStyle != GLU_POINT) {
	    needCache3 = 1;
	}
	if (qobj->drawStyle == GLU_LINE) {
	    needCache2 = 1;
	}
    }

    zNormal = deltaRadius / length;
    xyNormalRatio = height / length;

    for (i = 0; i < slices; i++) {
	angle = 2 * PI * i / slices;
	if (needCache2) {
	    if (qobj->orientation == GLU_OUTSIDE) {
		sinCache2[i] = xyNormalRatio * sin(angle);
		cosCache2[i] = xyNormalRatio * cos(angle);
	    } else {
		sinCache2[i] = -xyNormalRatio * sin(angle);
		cosCache2[i] = -xyNormalRatio * cos(angle);
	    }
	}
	sinCache[i] = sin(angle);
	cosCache[i] = cos(angle);
    }

    if (needCache3) {
	for (i = 0; i < slices; i++) {
	    angle = 2 * PI * (i-0.5) / slices;
	    if (qobj->orientation == GLU_OUTSIDE) {
		sinCache3[i] = xyNormalRatio * sin(angle);
		cosCache3[i] = xyNormalRatio * cos(angle);
	    } else {
		sinCache3[i] = -xyNormalRatio * sin(angle);
		cosCache3[i] = -xyNormalRatio * cos(angle);
	    }
	}
    }

    sinCache[slices] = sinCache[0];
    cosCache[slices] = cosCache[0];
    if (needCache2) {
	sinCache2[slices] = sinCache2[0];
	cosCache2[slices] = cosCache2[0];
    }
    if (needCache3) {
	sinCache3[slices] = sinCache3[0];
	cosCache3[slices] = cosCache3[0];
    }

    switch (qobj->drawStyle) {
      case GLU_FILL:
	/* Note:
	** An argument could be made for using a TRIANGLE_FAN for the end
	** of the cylinder of either radii is 0.0 (a cone).  However, a
	** TRIANGLE_FAN would not work in smooth shading mode (the common
	** case) because the normal for the apex is different for every
	** triangle (and TRIANGLE_FAN doesn't let me respecify that normal).
	** Now, my choice is GL_TRIANGLES, or leave the GL_QUAD_STRIP and
	** just let the GL trivially reject one of the two triangles of the
	** QUAD.  GL_QUAD_STRIP is probably faster, so I will leave this code
	** alone.
	*/
	for (j = 0; j < stacks; j++) {
	    zLow = j * height / stacks;
	    zHigh = (j + 1) * height / stacks;
	    radiusLow = baseRadius - deltaRadius * ((float) j / stacks);
	    radiusHigh = baseRadius - deltaRadius * ((float) (j + 1) / stacks);

	    glBegin(GL_QUAD_STRIP);
	    for (i = 0; i <= slices; i++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		    glNormal3f(sinCache3[i], cosCache3[i], zNormal);
		    break;
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2[i], cosCache2[i], zNormal);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->orientation == GLU_OUTSIDE) {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				(float) j / stacks);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], zLow);
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				(float) (j+1) / stacks);
		    }
		    glVertex3f(radiusHigh * sinCache[i],
			    radiusHigh * cosCache[i], zHigh);
		} else {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				(float) (j+1) / stacks);
		    }
		    glVertex3f(radiusHigh * sinCache[i],
			    radiusHigh * cosCache[i], zHigh);
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				(float) j / stacks);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], zLow);
		}
	    }
	    glEnd();
	}
	break;
      case GLU_POINT:
	glBegin(GL_POINTS);
	for (i = 0; i < slices; i++) {
	    switch(qobj->normals) {
	      case GLU_FLAT:
	      case GLU_SMOOTH:
		glNormal3f(sinCache2[i], cosCache2[i], zNormal);
		break;
	      case GLU_NONE:
	      default:
		break;
	    }
	    sintemp = sinCache[i];
	    costemp = cosCache[i];
	    for (j = 0; j <= stacks; j++) {
		zLow = j * height / stacks;
		radiusLow = baseRadius - deltaRadius * ((float) j / stacks);

		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    (float) j / stacks);
		}
		glVertex3f(radiusLow * sintemp,
			radiusLow * costemp, zLow);
	    }
	}
	glEnd();
	break;
      case GLU_LINE:
	for (j = 1; j < stacks; j++) {
	    zLow = j * height / stacks;
	    radiusLow = baseRadius - deltaRadius * ((float) j / stacks);

	    glBegin(GL_LINE_STRIP);
	    for (i = 0; i <= slices; i++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		    glNormal3f(sinCache3[i], cosCache3[i], zNormal);
		    break;
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2[i], cosCache2[i], zNormal);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    (float) j / stacks);
		}
		glVertex3f(radiusLow * sinCache[i],
			radiusLow * cosCache[i], zLow);
	    }
	    glEnd();
	}
	/* Intentionally fall through here... */
      case GLU_SILHOUETTE:
	for (j = 0; j <= stacks; j += stacks) {
	    zLow = j * height / stacks;
	    radiusLow = baseRadius - deltaRadius * ((float) j / stacks);

	    glBegin(GL_LINE_STRIP);
	    for (i = 0; i <= slices; i++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		    glNormal3f(sinCache3[i], cosCache3[i], zNormal);
		    break;
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2[i], cosCache2[i], zNormal);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    (float) j / stacks);
		}
		glVertex3f(radiusLow * sinCache[i], radiusLow * cosCache[i],
			zLow);
	    }
	    glEnd();
	}
	for (i = 0; i < slices; i++) {
	    switch(qobj->normals) {
	      case GLU_FLAT:
	      case GLU_SMOOTH:
		glNormal3f(sinCache2[i], cosCache2[i], 0.0);
		break;
	      case GLU_NONE:
	      default:
		break;
	    }
	    sintemp = sinCache[i];
	    costemp = cosCache[i];
	    glBegin(GL_LINE_STRIP);
	    for (j = 0; j <= stacks; j++) {
		zLow = j * height / stacks;
		radiusLow = baseRadius - deltaRadius * ((float) j / stacks);

		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    (float) j / stacks);
		}
		glVertex3f(radiusLow * sintemp,
			radiusLow * costemp, zLow);
	    }
	    glEnd();
	}
	break;
      default:
	break;
    }
}

inline void gluPartialDisk(GLUquadric *qobj, GLfloat innerRadius,
		   GLfloat outerRadius, GLint slices, GLint loops,
		   GLfloat startAngle, GLfloat sweepAngle);

inline void gluDisk(GLUquadric *qobj, GLfloat innerRadius, GLfloat outerRadius,
	    GLint slices, GLint loops)
{
    gluPartialDisk(qobj, innerRadius, outerRadius, slices, loops, 0.0, 360.0);
}

inline void gluPartialDisk(GLUquadric *qobj, GLfloat innerRadius,
		   GLfloat outerRadius, GLint slices, GLint loops,
		   GLfloat startAngle, GLfloat sweepAngle)
{
    GLint i,j;
    GLfloat sinCache[CACHE_SIZE];
    GLfloat cosCache[CACHE_SIZE];
    GLfloat angle;
    GLfloat sintemp, costemp;
    GLfloat deltaRadius;
    GLfloat radiusLow, radiusHigh;
    GLfloat texLow = 0.0, texHigh = 0.0;
    GLfloat angleOffset;
    GLint slices2;
    GLint finish;

    if (slices >= CACHE_SIZE) slices = CACHE_SIZE-1;
    if (slices < 2 || loops < 1 || outerRadius <= 0.0 || innerRadius < 0.0 ||
	    innerRadius > outerRadius) {
	gluQuadricError(qobj, GLU_INVALID_VALUE);
	return;
    }

    if (sweepAngle < -360.0) sweepAngle = 360.0;
    if (sweepAngle > 360.0) sweepAngle = 360.0;
    if (sweepAngle < 0) {
	startAngle += sweepAngle;
	sweepAngle = -sweepAngle;
    }

    if (sweepAngle == 360.0) {
	slices2 = slices;
    } else {
	slices2 = slices + 1;
    }

    /* Compute length (needed for normal calculations) */
    deltaRadius = outerRadius - innerRadius;

    /* Cache is the vertex locations cache */

    angleOffset = startAngle / 180.0 * PI;
    for (i = 0; i <= slices; i++) {
	angle = angleOffset + ((PI * sweepAngle) / 180.0) * i / slices;
	sinCache[i] = sin(angle);
	cosCache[i] = cos(angle);
    }

    if (sweepAngle == 360.0) {
	sinCache[slices] = sinCache[0];
	cosCache[slices] = cosCache[0];
    }

    switch(qobj->normals) {
      case GLU_FLAT:
      case GLU_SMOOTH:
	if (qobj->orientation == GLU_OUTSIDE) {
	    glNormal3f(0.0, 0.0, 1.0);
	} else {
	    glNormal3f(0.0, 0.0, -1.0);
	}
	break;
      default:
      case GLU_NONE:
	break;
    }

    switch (qobj->drawStyle) {
      case GLU_FILL:
	if (innerRadius == 0.0) {
	    finish = loops - 1;
	    /* Triangle strip for inner polygons */
	    glBegin(GL_TRIANGLE_FAN);
	    if (qobj->textureCoords) {
		glTexCoord2f(0.5, 0.5);
	    }
	    glVertex3f(0.0, 0.0, 0.0);
	    radiusLow = outerRadius -
		    deltaRadius * ((float) (loops-1) / loops);
	    if (qobj->textureCoords) {
		texLow = radiusLow / outerRadius / 2;
	    }

	    if (qobj->orientation == GLU_OUTSIDE) {
		for (i = slices; i >= 0; i--) {
		    if (qobj->textureCoords) {
			glTexCoord2f(texLow * sinCache[i] + 0.5,
				texLow * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], 0.0);
		}
	    } else {
		for (i = 0; i <= slices; i++) {
		    if (qobj->textureCoords) {
			glTexCoord2f(texLow * sinCache[i] + 0.5,
				texLow * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], 0.0);
		}
	    }
	    glEnd();
	} else {
	    finish = loops;
	}
	for (j = 0; j < finish; j++) {
	    radiusLow = outerRadius - deltaRadius * ((float) j / loops);
	    radiusHigh = outerRadius - deltaRadius * ((float) (j + 1) / loops);
	    if (qobj->textureCoords) {
		texLow = radiusLow / outerRadius / 2;
		texHigh = radiusHigh / outerRadius / 2;
	    }

	    glBegin(GL_QUAD_STRIP);
	    for (i = 0; i <= slices; i++) {
		if (qobj->orientation == GLU_OUTSIDE) {
		    if (qobj->textureCoords) {
			glTexCoord2f(texLow * sinCache[i] + 0.5,
				texLow * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], 0.0);

		    if (qobj->textureCoords) {
			glTexCoord2f(texHigh * sinCache[i] + 0.5,
				texHigh * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusHigh * sinCache[i],
			    radiusHigh * cosCache[i], 0.0);
		} else {
		    if (qobj->textureCoords) {
			glTexCoord2f(texHigh * sinCache[i] + 0.5,
				texHigh * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusHigh * sinCache[i],
			    radiusHigh * cosCache[i], 0.0);

		    if (qobj->textureCoords) {
			glTexCoord2f(texLow * sinCache[i] + 0.5,
				texLow * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusLow * sinCache[i],
			    radiusLow * cosCache[i], 0.0);
		}
	    }
	    glEnd();
	}
	break;
      case GLU_POINT:
	glBegin(GL_POINTS);
	for (i = 0; i < slices2; i++) {
	    sintemp = sinCache[i];
	    costemp = cosCache[i];
	    for (j = 0; j <= loops; j++) {
		radiusLow = outerRadius - deltaRadius * ((float) j / loops);

		if (qobj->textureCoords) {
		    texLow = radiusLow / outerRadius / 2;

		    glTexCoord2f(texLow * sinCache[i] + 0.5,
			    texLow * cosCache[i] + 0.5);
		}
		glVertex3f(radiusLow * sintemp, radiusLow * costemp, 0.0);
	    }
	}
	glEnd();
	break;
      case GLU_LINE:
	if (innerRadius == outerRadius) {
	    glBegin(GL_LINE_STRIP);

	    for (i = 0; i <= slices; i++) {
		if (qobj->textureCoords) {
		    glTexCoord2f(sinCache[i] / 2 + 0.5,
			    cosCache[i] / 2 + 0.5);
		}
		glVertex3f(innerRadius * sinCache[i],
			innerRadius * cosCache[i], 0.0);
	    }
	    glEnd();
	    break;
	}
	for (j = 0; j <= loops; j++) {
	    radiusLow = outerRadius - deltaRadius * ((float) j / loops);
	    if (qobj->textureCoords) {
		texLow = radiusLow / outerRadius / 2;
	    }

	    glBegin(GL_LINE_STRIP);
	    for (i = 0; i <= slices; i++) {
		if (qobj->textureCoords) {
		    glTexCoord2f(texLow * sinCache[i] + 0.5,
			    texLow * cosCache[i] + 0.5);
		}
		glVertex3f(radiusLow * sinCache[i],
			radiusLow * cosCache[i], 0.0);
	    }
	    glEnd();
	}
	for (i=0; i < slices2; i++) {
	    sintemp = sinCache[i];
	    costemp = cosCache[i];
	    glBegin(GL_LINE_STRIP);
	    for (j = 0; j <= loops; j++) {
		radiusLow = outerRadius - deltaRadius * ((float) j / loops);
		if (qobj->textureCoords) {
		    texLow = radiusLow / outerRadius / 2;
		}

		if (qobj->textureCoords) {
		    glTexCoord2f(texLow * sinCache[i] + 0.5,
			    texLow * cosCache[i] + 0.5);
		}
		glVertex3f(radiusLow * sintemp, radiusLow * costemp, 0.0);
	    }
	    glEnd();
	}
	break;
      case GLU_SILHOUETTE:
	if (sweepAngle < 360.0) {
	    for (i = 0; i <= slices; i+= slices) {
		sintemp = sinCache[i];
		costemp = cosCache[i];
		glBegin(GL_LINE_STRIP);
		for (j = 0; j <= loops; j++) {
		    radiusLow = outerRadius - deltaRadius * ((float) j / loops);

		    if (qobj->textureCoords) {
			texLow = radiusLow / outerRadius / 2;
			glTexCoord2f(texLow * sinCache[i] + 0.5,
				texLow * cosCache[i] + 0.5);
		    }
		    glVertex3f(radiusLow * sintemp, radiusLow * costemp, 0.0);
		}
		glEnd();
	    }
	}
	for (j = 0; j <= loops; j += loops) {
	    radiusLow = outerRadius - deltaRadius * ((float) j / loops);
	    if (qobj->textureCoords) {
		texLow = radiusLow / outerRadius / 2;
	    }

	    glBegin(GL_LINE_STRIP);
	    for (i = 0; i <= slices; i++) {
		if (qobj->textureCoords) {
		    glTexCoord2f(texLow * sinCache[i] + 0.5,
			    texLow * cosCache[i] + 0.5);
		}
		glVertex3f(radiusLow * sinCache[i],
			radiusLow * cosCache[i], 0.0);
	    }
	    glEnd();
	    if (innerRadius == outerRadius) break;
	}
	break;
      default:
	break;
    }
}

inline void gluSphere(GLUquadric *qobj, GLfloat radius, GLint slices, GLint stacks)
{
    GLint i,j;
    GLfloat sinCache1a[CACHE_SIZE];
    GLfloat cosCache1a[CACHE_SIZE];
    GLfloat sinCache2a[CACHE_SIZE];
    GLfloat cosCache2a[CACHE_SIZE];
    GLfloat sinCache3a[CACHE_SIZE];
    GLfloat cosCache3a[CACHE_SIZE];
    GLfloat sinCache1b[CACHE_SIZE];
    GLfloat cosCache1b[CACHE_SIZE];
    GLfloat sinCache2b[CACHE_SIZE];
    GLfloat cosCache2b[CACHE_SIZE];
    GLfloat sinCache3b[CACHE_SIZE];
    GLfloat cosCache3b[CACHE_SIZE];
    GLfloat angle;
    GLfloat zLow, zHigh;
    GLfloat sintemp1 = 0.0, sintemp2 = 0.0, sintemp3 = 0.0, sintemp4 = 0.0;
    GLfloat costemp1 = 0.0, costemp2 = 0.0, costemp3 = 0.0, costemp4 = 0.0;
    GLboolean needCache2, needCache3;
    GLint start, finish;

    if (slices >= CACHE_SIZE) slices = CACHE_SIZE-1;
    if (stacks >= CACHE_SIZE) stacks = CACHE_SIZE-1;
    if (slices < 2 || stacks < 1 || radius < 0.0) {
	gluQuadricError(qobj, GLU_INVALID_VALUE);
	return;
    }

    /* Cache is the vertex locations cache */
    /* Cache2 is the various normals at the vertices themselves */
    /* Cache3 is the various normals for the faces */
    needCache2 = needCache3 = GL_FALSE;

    if (qobj->normals == GLU_SMOOTH) {
	needCache2 = GL_TRUE;
    }

    if (qobj->normals == GLU_FLAT) {
	if (qobj->drawStyle != GLU_POINT) {
	    needCache3 = GL_TRUE;
	}
	if (qobj->drawStyle == GLU_LINE) {
	    needCache2 = GL_TRUE;
	}
    }

    for (i = 0; i < slices; i++) {
	angle = 2 * PI * i / slices;
	sinCache1a[i] = sin(angle);
	cosCache1a[i] = cos(angle);
	if (needCache2) {
	    sinCache2a[i] = sinCache1a[i];
	    cosCache2a[i] = cosCache1a[i];
	}
    }

    for (j = 0; j <= stacks; j++) {
	angle = PI * j / stacks;
	if (needCache2) {
	    if (qobj->orientation == GLU_OUTSIDE) {
		sinCache2b[j] = sin(angle);
		cosCache2b[j] = cos(angle);
	    } else {
		sinCache2b[j] = -sin(angle);
		cosCache2b[j] = -cos(angle);
	    }
	}
	sinCache1b[j] = radius * sin(angle);
	cosCache1b[j] = radius * cos(angle);
    }
    /* Make sure it comes to a point */
    sinCache1b[0] = 0;
    sinCache1b[stacks] = 0;

    if (needCache3) {
	for (i = 0; i < slices; i++) {
	    angle = 2 * PI * (i-0.5) / slices;
	    sinCache3a[i] = sin(angle);
	    cosCache3a[i] = cos(angle);
	}
	for (j = 0; j <= stacks; j++) {
	    angle = PI * (j - 0.5) / stacks;
	    if (qobj->orientation == GLU_OUTSIDE) {
		sinCache3b[j] = sin(angle);
		cosCache3b[j] = cos(angle);
	    } else {
		sinCache3b[j] = -sin(angle);
		cosCache3b[j] = -cos(angle);
	    }
	}
    }

    sinCache1a[slices] = sinCache1a[0];
    cosCache1a[slices] = cosCache1a[0];
    if (needCache2) {
	sinCache2a[slices] = sinCache2a[0];
	cosCache2a[slices] = cosCache2a[0];
    }
    if (needCache3) {
	sinCache3a[slices] = sinCache3a[0];
	cosCache3a[slices] = cosCache3a[0];
    }

    switch (qobj->drawStyle) {
      case GLU_FILL:
	/* Do ends of sphere as TRIANGLE_FAN's (if not texturing)
	** We don't do it when texturing because we need to respecify the
	** texture coordinates of the apex for every adjacent vertex (because
	** it isn't a constant for that point)
	*/
	if (!(qobj->textureCoords)) {
	    start = 1;
	    finish = stacks - 1;

	    /* Low end first (j == 0 iteration) */
	    sintemp2 = sinCache1b[1];
	    zHigh = cosCache1b[1];
	    switch(qobj->normals) {
	      case GLU_FLAT:
		sintemp3 = sinCache3b[1];
		costemp3 = cosCache3b[1];
		break;
	      case GLU_SMOOTH:
		sintemp3 = sinCache2b[1];
		costemp3 = cosCache2b[1];
		glNormal3f(sinCache2a[0] * sinCache2b[0],
			cosCache2a[0] * sinCache2b[0],
			cosCache2b[0]);
		break;
	      default:
		break;
	    }
	    glBegin(GL_TRIANGLE_FAN);
	    glVertex3f(0.0, 0.0, radius);
	    if (qobj->orientation == GLU_OUTSIDE) {
		for (i = slices; i >= 0; i--) {
		    switch(qobj->normals) {
		      case GLU_SMOOTH:
			glNormal3f(sinCache2a[i] * sintemp3,
				cosCache2a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_FLAT:
			if (i != slices) {
			    glNormal3f(sinCache3a[i+1] * sintemp3,
				    cosCache3a[i+1] * sintemp3,
				    costemp3);
			}
			break;
		      case GLU_NONE:
		      default:
			break;
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		}
	    } else {
		for (i = 0; i <= slices; i++) {
		    switch(qobj->normals) {
		      case GLU_SMOOTH:
			glNormal3f(sinCache2a[i] * sintemp3,
				cosCache2a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_FLAT:
			glNormal3f(sinCache3a[i] * sintemp3,
				cosCache3a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_NONE:
		      default:
			break;
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		}
	    }
	    glEnd();

	    /* High end next (j == stacks-1 iteration) */
	    sintemp2 = sinCache1b[stacks-1];
	    zHigh = cosCache1b[stacks-1];
	    switch(qobj->normals) {
	      case GLU_FLAT:
		sintemp3 = sinCache3b[stacks];
		costemp3 = cosCache3b[stacks];
		break;
	      case GLU_SMOOTH:
		sintemp3 = sinCache2b[stacks-1];
		costemp3 = cosCache2b[stacks-1];
		glNormal3f(sinCache2a[stacks] * sinCache2b[stacks],
			cosCache2a[stacks] * sinCache2b[stacks],
			cosCache2b[stacks]);
		break;
	      default:
		break;
	    }
	    glBegin(GL_TRIANGLE_FAN);
	    glVertex3f(0.0, 0.0, -radius);
	    if (qobj->orientation == GLU_OUTSIDE) {
		for (i = 0; i <= slices; i++) {
		    switch(qobj->normals) {
		      case GLU_SMOOTH:
			glNormal3f(sinCache2a[i] * sintemp3,
				cosCache2a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_FLAT:
			glNormal3f(sinCache3a[i] * sintemp3,
				cosCache3a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_NONE:
		      default:
			break;
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		}
	    } else {
		for (i = slices; i >= 0; i--) {
		    switch(qobj->normals) {
		      case GLU_SMOOTH:
			glNormal3f(sinCache2a[i] * sintemp3,
				cosCache2a[i] * sintemp3,
				costemp3);
			break;
		      case GLU_FLAT:
			if (i != slices) {
			    glNormal3f(sinCache3a[i+1] * sintemp3,
				    cosCache3a[i+1] * sintemp3,
				    costemp3);
			}
			break;
		      case GLU_NONE:
		      default:
			break;
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		}
	    }
	    glEnd();
	} else {
	    start = 0;
	    finish = stacks;
	}
	for (j = start; j < finish; j++) {
	    zLow = cosCache1b[j];
	    zHigh = cosCache1b[j+1];
	    sintemp1 = sinCache1b[j];
	    sintemp2 = sinCache1b[j+1];
	    switch(qobj->normals) {
	      case GLU_FLAT:
		sintemp4 = sinCache3b[j+1];
		costemp4 = cosCache3b[j+1];
		break;
	      case GLU_SMOOTH:
		if (qobj->orientation == GLU_OUTSIDE) {
		    sintemp3 = sinCache2b[j+1];
		    costemp3 = cosCache2b[j+1];
		    sintemp4 = sinCache2b[j];
		    costemp4 = cosCache2b[j];
		} else {
		    sintemp3 = sinCache2b[j];
		    costemp3 = cosCache2b[j];
		    sintemp4 = sinCache2b[j+1];
		    costemp4 = cosCache2b[j+1];
		}
		break;
	      default:
		break;
	    }

	    glBegin(GL_QUAD_STRIP);
	    for (i = 0; i <= slices; i++) {
		switch(qobj->normals) {
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2a[i] * sintemp3,
			    cosCache2a[i] * sintemp3,
			    costemp3);
		    break;
		  case GLU_FLAT:
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->orientation == GLU_OUTSIDE) {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				1 - (float) (j+1) / stacks);
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		} else {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				1 - (float) j / stacks);
		    }
		    glVertex3f(sintemp1 * sinCache1a[i],
			    sintemp1 * cosCache1a[i], zLow);
		}
		switch(qobj->normals) {
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2a[i] * sintemp4,
			    cosCache2a[i] * sintemp4,
			    costemp4);
		    break;
		  case GLU_FLAT:
		    glNormal3f(sinCache3a[i] * sintemp4,
			    cosCache3a[i] * sintemp4,
			    costemp4);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->orientation == GLU_OUTSIDE) {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				1 - (float) j / stacks);
		    }
		    glVertex3f(sintemp1 * sinCache1a[i],
			    sintemp1 * cosCache1a[i], zLow);
		} else {
		    if (qobj->textureCoords) {
			glTexCoord2f(1 - (float) i / slices,
				1 - (float) (j+1) / stacks);
		    }
		    glVertex3f(sintemp2 * sinCache1a[i],
			    sintemp2 * cosCache1a[i], zHigh);
		}
	    }
	    glEnd();
	}
	break;
      case GLU_POINT:
	glBegin(GL_POINTS);
	for (j = 0; j <= stacks; j++) {
	    sintemp1 = sinCache1b[j];
	    costemp1 = cosCache1b[j];
	    switch(qobj->normals) {
	      case GLU_FLAT:
	      case GLU_SMOOTH:
		sintemp2 = sinCache2b[j];
		costemp2 = cosCache2b[j];
		break;
	      default:
		break;
	    }
	    for (i = 0; i < slices; i++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2a[i] * sintemp2,
			    cosCache2a[i] * sintemp2,
			    costemp2);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}

		zLow = j * radius / stacks;

		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    1 - (float) j / stacks);
		}
		glVertex3f(sintemp1 * sinCache1a[i],
			sintemp1 * cosCache1a[i], costemp1);
	    }
	}
	glEnd();
	break;
      case GLU_LINE:
      case GLU_SILHOUETTE:
	for (j = 1; j < stacks; j++) {
	    sintemp1 = sinCache1b[j];
	    costemp1 = cosCache1b[j];
	    switch(qobj->normals) {
	      case GLU_FLAT:
	      case GLU_SMOOTH:
		sintemp2 = sinCache2b[j];
		costemp2 = cosCache2b[j];
		break;
	      default:
		break;
	    }

	    glBegin(GL_LINE_STRIP);
	    for (i = 0; i <= slices; i++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		    glNormal3f(sinCache3a[i] * sintemp2,
			    cosCache3a[i] * sintemp2,
			    costemp2);
		    break;
		  case GLU_SMOOTH:
		    glNormal3f(sinCache2a[i] * sintemp2,
			    cosCache2a[i] * sintemp2,
			    costemp2);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}
		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    1 - (float) j / stacks);
		}
		glVertex3f(sintemp1 * sinCache1a[i],
			sintemp1 * cosCache1a[i], costemp1);
	    }
	    glEnd();
	}
	for (i = 0; i < slices; i++) {
	    sintemp1 = sinCache1a[i];
	    costemp1 = cosCache1a[i];
	    switch(qobj->normals) {
	      case GLU_FLAT:
	      case GLU_SMOOTH:
		sintemp2 = sinCache2a[i];
		costemp2 = cosCache2a[i];
		break;
	      default:
		break;
	    }

	    glBegin(GL_LINE_STRIP);
	    for (j = 0; j <= stacks; j++) {
		switch(qobj->normals) {
		  case GLU_FLAT:
		    glNormal3f(sintemp2 * sinCache3b[j],
			    costemp2 * sinCache3b[j],
			    cosCache3b[j]);
		    break;
		  case GLU_SMOOTH:
		    glNormal3f(sintemp2 * sinCache2b[j],
			    costemp2 * sinCache2b[j],
			    cosCache2b[j]);
		    break;
		  case GLU_NONE:
		  default:
		    break;
		}

		if (qobj->textureCoords) {
		    glTexCoord2f(1 - (float) i / slices,
			    1 - (float) j / stacks);
		}
		glVertex3f(sintemp1 * sinCache1b[j],
			costemp1 * sinCache1b[j], cosCache1b[j]);
	    }
	    glEnd();
	}
	break;
      default:
	break;
    }
}

#endif // _ps3gl_compat_h_
