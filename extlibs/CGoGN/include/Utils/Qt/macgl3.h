//#include <QApplication>
#include <QGLWidget>
#include <QGLContext>
#include <iostream>

//external Objective C function
void* select_3_2_mac_visual(GDHandle handle);

namespace CGoGN
{

/// Use with  GLWidget(...):QGLWidget(new Core3_2_context(QGLFormat::defaultFormat()))
/// check with std::cout<<glGetString(GL_VERSION)<<std::endl;
struct Core3_2_context : public QGLContext
{
	Core3_2_context(const QGLFormat& format, QPaintDevice* device) : QGLContext(format,device) {}
	Core3_2_context(const QGLFormat& format) : QGLContext(format) {}

	virtual void* chooseMacVisual(GDHandle handle)
	{
		return select_3_2_mac_visual(handle);
	}
};

}
