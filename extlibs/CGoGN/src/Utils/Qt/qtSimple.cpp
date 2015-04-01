/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/
#define CGoGN_UTILS_DLL_EXPORT 1
#include <cmath>
#include "Utils/GLSLShader.h"
#include "Utils/Qt/qtSimple.h"
#include "Utils/Qt/qtgl.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_precision.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <QTextEdit>
#include <QImage>
#include <QMenuBar>
#include <QAction>
#include <QDockWidget>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>

namespace CGoGN
{

namespace Utils
{

namespace QT
{

SimpleQT::SimpleQT() :
	m_dock(NULL),
	m_projection_matrix(m_mat.m_matrices[0]),
	m_modelView_matrix(m_mat.m_matrices[1]),
	m_transfo_matrix(m_mat.m_matrices[2])
{
	if (GLSLShader::CURRENT_OGL_VERSION >= 3)
	{
		QGLFormat glFormat;
		glFormat.setVersion( Utils::GLSLShader::MAJOR_OGL_CORE, Utils::GLSLShader::MINOR_OGL_CORE);
		glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
		glFormat.setSampleBuffers( true );
		QGLFormat::setDefaultFormat(glFormat);
	}
	m_glWidget = new GLWidget(this);


	setCentralWidget(m_glWidget);
	setWindowTitle(tr("CGoGN"));

	m_fileMenu = menuBar()->addMenu(tr("&File"));

	QAction* action = new QAction(tr("New"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_New()));
	m_fileMenu->addAction(action);

	action = new QAction(tr("Open"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_Open()));
	m_fileMenu->addAction(action);

	action = new QAction(tr("Save"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_Save()));
	m_fileMenu->addAction(action);

	m_fileMenu->addSeparator();

	action = new QAction(tr("Quit"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_Quit()));
	m_fileMenu->addAction(action);

	m_appMenu = menuBar()->addMenu(tr("&Application"));

	QMenu* m_helpMenu = menuBar()->addMenu(tr("&Help"));

	action = new QAction(tr("console on/off"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_consoleOnOff()));
	m_helpMenu->addAction(action);

	action = new QAction(tr("console clear"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_consoleClear()));
	m_helpMenu->addAction(action);

	action = new QAction(tr("About"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_about()));
	m_helpMenu->addAction(action);

	action = new QAction(tr("About CGoGN"), this);
	connect(action, SIGNAL(triggered()), this, SLOT(cb_about_cgogn()));
	m_helpMenu->addAction(action);

	m_dockConsole = new QDockWidget(tr("Console"), this);
	m_dockConsole->setAllowedAreas(Qt::BottomDockWidgetArea);
	m_dockConsole->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	addDockWidget(Qt::BottomDockWidgetArea, m_dockConsole);

	m_textConsole = new QTextEdit();
	m_textConsole->setLineWrapMode(QTextEdit::NoWrap);
	m_textConsole->setTabStopWidth(20);
//	m_textConsole->setReadOnly(true);

	m_dockConsole->setWidget(m_textConsole);

	m_dockConsole->hide();

	m_transfo_matrix = glm::mat4(1.0f);

	resize(1200,800);
	m_glWidget->setFocus(Qt::MouseFocusReason);
}

SimpleQT::SimpleQT(const SimpleQT& sqt):
	QMainWindow(),
	m_dock(NULL),
	m_projection_matrix(m_mat.m_matrices[0]),
	m_modelView_matrix(m_mat.m_matrices[1]),
	m_transfo_matrix(m_mat.m_matrices[2])
{
	if (GLSLShader::CURRENT_OGL_VERSION >= 3)
	{
		QGLFormat glFormat;
		glFormat.setVersion( Utils::GLSLShader::MAJOR_OGL_CORE, Utils::GLSLShader::MINOR_OGL_CORE);
		glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
		glFormat.setSampleBuffers( true );
		QGLFormat::setDefaultFormat(glFormat);
	}

	m_glWidget = new GLWidget(this);

	setCentralWidget(m_glWidget);

	m_dock = new QDockWidget(sqt.m_dock) ;
	m_dockConsole = new QDockWidget(sqt.m_dockConsole) ;
	m_textConsole = new QTextEdit(sqt.m_textConsole) ;
	m_dockOn = sqt.m_dockOn ;


	for (unsigned int i = 0; i < 4; ++i)
	{
		m_curquat[i] = sqt.m_curquat[i];
		m_lastquat[i] = sqt.m_lastquat[i];
	}
	m_trans_x = sqt.m_trans_x ;
	m_trans_y = sqt.m_trans_y ;
	m_trans_z = sqt.m_trans_z ;

	m_glWidget->setFocus(Qt::MouseFocusReason);
}

SimpleQT::~SimpleQT()
{
	delete m_glWidget; // ??
}

void SimpleQT::operator=(const SimpleQT& sqt)
{
	if (GLSLShader::CURRENT_OGL_VERSION >= 3)
	{
		QGLFormat glFormat;
		glFormat.setVersion( Utils::GLSLShader::MAJOR_OGL_CORE, Utils::GLSLShader::MINOR_OGL_CORE);
		glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
		glFormat.setSampleBuffers( true );
		QGLFormat::setDefaultFormat(glFormat);
	}

	m_glWidget = new GLWidget(this);

	setCentralWidget(m_glWidget) ;

	m_dock = new QDockWidget(sqt.m_dock) ;
	m_dockConsole = new QDockWidget(sqt.m_dockConsole) ;
	m_textConsole = new QTextEdit(sqt.m_textConsole) ;
	m_dockOn = sqt.m_dockOn ;

	m_projection_matrix = sqt.m_projection_matrix;
	m_modelView_matrix = sqt.m_modelView_matrix;
	for (unsigned int i = 0; i < 4; ++i)
	{
		m_curquat[i] = sqt.m_curquat[i];
		m_lastquat[i] = sqt.m_lastquat[i];
	}
	m_trans_x = sqt.m_trans_x ;
	m_trans_y = sqt.m_trans_y ;
	m_trans_z = sqt.m_trans_z ;
}

void SimpleQT::setDock(QDockWidget *dock)
{
	m_dock = dock;
	addDockWidget(Qt::RightDockWidgetArea, m_dock);
	m_dock->show();
}

QDockWidget* SimpleQT::dockWidget()
{
	return m_dock;
}

void SimpleQT::setCallBack( const QObject* sender, const char* signal, const char* method)
{
	connect(sender, signal, this, method);
}

void SimpleQT::windowTitle(const char* windowTitle)
{
	setWindowTitle(tr(windowTitle));
}

void SimpleQT::dockTitle(const char* dockTitle)
{
	if (m_dock)
		m_dock->setWindowTitle(tr(dockTitle));
}

void SimpleQT::statusMsg(const char* msg, int timeoutms)
{
	if (msg)
	{
		QString message = tr(msg);
		statusBar()->showMessage(message,timeoutms);
	}
	else
	{
		if (statusBar()->isHidden())
			statusBar()->show();
		else
			statusBar()->hide();
	}
}

QDockWidget* SimpleQT::addEmptyDock()
{
	m_dock = new QDockWidget(tr("Control"), this);
	m_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
	m_dock->setFeatures(QDockWidget::DockWidgetMovable|QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetClosable);
	addDockWidget(Qt::RightDockWidgetArea, m_dock);

	m_dock->hide();
	return m_dock;
}

void SimpleQT::visibilityDock(bool visible)
{
	if (visible)
		m_dock->show();
	else
		m_dock->hide();
}

void SimpleQT::visibilityConsole(bool visible)
{
	if (visible)
		m_dockConsole->show();
	else
		m_dockConsole->hide();
}

void SimpleQT::toggleVisibilityDock()
{
	if (m_dock->isHidden())
		m_dock->show();
	else
		m_dock->hide();
}

void SimpleQT::toggleVisibilityConsole()
{
	if (m_dockConsole->isHidden())
		m_dockConsole->show();
	else
		m_dockConsole->hide();
}

void SimpleQT::add_menu_entry(const std::string label, const char* method)
{
	QAction * action = new QAction(tr(label.c_str()), this);
	connect(action, SIGNAL(triggered()), this, method);
	m_appMenu->addAction(action);
}

void SimpleQT::init_app_menu()
{
	m_appMenu->clear();
}

void SimpleQT::setHelpMsg(const std::string& msg)
{
	m_helpString = msg;
}

void SimpleQT::setGLWidgetMouseTracking(bool b)
{
	m_glWidget->setMouseTracking(b);
}

void SimpleQT::closeEvent(QCloseEvent *event)
{
	QWidget::closeEvent(event) ;
	cb_exit();
}

void SimpleQT::keyPressEvent(QKeyEvent *e)
{
	if (e->modifiers() & Qt::ShiftModifier)
	{
		if ((e->key() == Qt::Key_Return))
			toggleVisibilityConsole();
	}
	else
	{
		if ((e->key() == Qt::Key_Return) && m_dock != NULL)
			toggleVisibilityDock();
	}

    if (e->key() == Qt::Key_Escape)
    	close();
    else
        QWidget::keyPressEvent(e);

    m_glWidget->keyPressEvent(e); // ?
}

void SimpleQT::keyReleaseEvent(QKeyEvent *e)
{
	QWidget::keyReleaseEvent(e);
    m_glWidget->keyReleaseEvent(e);
}

void SimpleQT::glMousePosition(int& x, int& y)
{
	QPoint xy = m_glWidget->mapFromGlobal(QCursor::pos());
	x = xy.x();
	y = m_glWidget->getHeight() - xy.y();
}




void SimpleQT::synchronize(SimpleQT* sqt)
{
	m_glWidget->getObjPos() = sqt->m_glWidget->getObjPos() ;

	m_projection_matrix = sqt->m_projection_matrix;
	m_modelView_matrix = sqt->m_modelView_matrix;
	for (unsigned int i = 0; i < 4; ++i)
	{
		m_curquat[i] = sqt->m_curquat[i];
		m_lastquat[i] = sqt->m_lastquat[i];
	}
	m_trans_x = sqt->trans_x();
	m_trans_y = sqt->trans_y();
	m_trans_z = sqt->trans_z();

	SimpleQT::cb_updateMatrix();

	m_glWidget->modelModified();
	m_glWidget->updateGL();
}

void SimpleQT::registerShader(GLSLShader* ptr)
{
	GLSLShader::registerShader(this, ptr) ;
}

void SimpleQT::unregisterShader(GLSLShader* ptr)
{
	GLSLShader::unregisterShader(this, ptr) ;
}

void SimpleQT::cb_updateMatrix()
{
	glm::mat4 model(m_modelView_matrix);
	model *= m_transfo_matrix;

	if (GLSLShader::CURRENT_OGL_VERSION == 1)
	{
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(glm::value_ptr(m_projection_matrix));

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(glm::value_ptr(model));
	}
	else
	{
		for(std::set< std::pair<void*, GLSLShader*> >::iterator it = GLSLShader::m_registeredShaders->begin();
			it != GLSLShader::m_registeredShaders->end();
			++it)
		{
			if ((it->first == NULL) || (it->first == this))
			{
				it->second->updateMatrices(m_projection_matrix, model);
			}
		}
	}
}

void SimpleQT::updateGL()
{
	m_glWidget->updateGL();
}

void SimpleQT::updateGLMatrices()
{
	m_glWidget->modelModified();
    updateGL();
}

void SimpleQT::transfoRotate(float angle, float x, float y, float z)
{
	transfoMatrix() = glm::rotate(transfoMatrix(), glm::radians(angle), glm::vec3(x,y,z));
}

void SimpleQT::transfoTranslate(float tx, float ty, float tz)
{
	transfoMatrix() = glm::translate(transfoMatrix(), glm::vec3(tx,ty,tz));
}

void SimpleQT::transfoScale(float sx, float sy, float sz)
{
	transfoMatrix() = glm::scale(transfoMatrix(), glm::vec3(sx,sy,sz));
}

void SimpleQT::pushTransfoMatrix()
{
	m_stack_trf.push(transfoMatrix());
}

bool SimpleQT::popTransfoMatrix()
{
	if (m_stack_trf.empty())
		return false;
	transfoMatrix() = m_stack_trf.top();
	m_stack_trf.pop();
	return true;
}

std::string SimpleQT::selectFile(const std::string& title, const std::string& dir, const std::string& filters)
{
    QString fileName = QFileDialog::getOpenFileName(this, tr(title.c_str()), tr(dir.c_str()), tr(filters.c_str()), 0, 0);
    return fileName.toStdString();
}

std::string SimpleQT::selectFileSave(const std::string& title, const std::string& dir, const std::string& filters)
{
    QString fileName = QFileDialog::getSaveFileName(this, tr(title.c_str()), tr(dir.c_str()), tr(filters.c_str()), 0, 0);
    return fileName.toStdString();
}

void SimpleQT::cb_about_cgogn()
{
	QString str("CGoGN:\nCombinatorial and Geometric modeling\n"
				"with Generic N-dimensional Maps\n"
				"Web site: http://cgogn.unistra.fr \n"
				"Contact information: cgogn@unistra.fr");
	QMessageBox::about(this, tr("About CGoGN"), str);
}

void SimpleQT::cb_about()
{
   QMessageBox::about(this, tr("About App"), m_helpString.c_str());
}

void SimpleQT::snapshot(const QString& filename, const char* format, const int& quality)
{
	QImage im = m_glWidget->grabFrameBuffer(false);
	im.save(filename, format, quality);
}

void SimpleQT::exportPOV2file(const QString& filename)
{
    std::ofstream out(filename.toStdString().c_str(), std::ios::out) ;
    if (!out.good())
    {
        CGoGNerr << "Unable to open file" << CGoGNendl ;
    }

    out << m_glWidget->getObjPos().x << std::endl ;
    out << m_glWidget->getObjPos().y << std::endl ;
    out << m_glWidget->getObjPos().z << std::endl ;

    for (unsigned int i = 0 ; i < 4 ; ++i)
        for (unsigned int j = 0 ; j < 4 ; ++j)
            out << m_projection_matrix[i][j] << std::endl ;

    for (unsigned int i = 0 ; i < 4 ; ++i)
        for (unsigned int j = 0 ; j < 4 ; ++j)
            out << m_modelView_matrix[i][j] << std::endl ;

    for (unsigned int i = 0; i < 4; ++i)
    {
        out << m_curquat[i] << std::endl ;
        out << m_lastquat[i] << std::endl ;
    }

    out << m_trans_x << std::endl ;
    out << m_trans_y << std::endl ;
    out << m_trans_z << std::endl ;

    QRect geom = this->geometry() ;
    out << geom.width() << std::endl ;
    out << geom.height() << std::endl ;
}

void SimpleQT::importFile2POV(const QString& filename)
{
    std::ifstream in(filename.toStdString().c_str(), std::ios::in) ;
    if (!in.good())
    {
        CGoGNerr << "Unable to open file" << CGoGNendl ;
    }

    in >> m_glWidget->getObjPos().x ;
    in >> m_glWidget->getObjPos().y ;
    in >> m_glWidget->getObjPos().z ;

    for (unsigned int i = 0 ; i < 4 ; ++i)
        for (unsigned int j = 0 ; j < 4 ; ++j)
            in >> m_projection_matrix[i][j] ;

    for (unsigned int i = 0 ; i < 4 ; ++i)
        for (unsigned int j = 0 ; j < 4 ; ++j)
            in >> m_modelView_matrix[i][j] ;

    for (unsigned int i = 0; i < 4; ++i)
    {
        in >> m_curquat[i] ;
        in >> m_lastquat[i] ;
    }

    in >> m_trans_x ;
    in >> m_trans_y ;
    in >> m_trans_z ;

    unsigned int width, height ;
    in >> width ;
    in >> height ;

    this->resize(width, height) ;

    SimpleQT::cb_updateMatrix() ;
    m_glWidget->modelModified() ;
    updateGL() ;
}


void SimpleQT::setGeometry(int x, int y, int w, int h)
{
	move(x,y);
	resize(w,h);
}

} // namespace QT

} // namespace Utils

} // namespace CGoGN
