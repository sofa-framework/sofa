/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "QtOgreViewer.h"
#include <sofa/gui/SofaGUI.h>
#include <sofa/gui/qt/GenGraphForm.h>

#ifdef SOFA_QT4
#include <QFileDialog>
#else
#include <qfiledialog.h>
#endif


namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{
namespace qtogre
{

//*****************************************************************************************
// Just modify the point of view, keep the simulation running.
void QtOgreViewer::resetView()
{
    using namespace Ogre;

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+"."+sofa::gui::SofaGUI::GetGUIName()+".view";
        std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
            Ogre::Vector3 camera_position;
            Ogre::Vector3 camera_zero;
            Ogre::Quaternion camera_orientation;

            in >> camera_zero[0];
            in >> camera_zero[1];
            in >> camera_zero[2];

            in >> camera_position[0];
            in >> camera_position[1];
            in >> camera_position[2];

            in >> camera_orientation.x;
            in >> camera_orientation.y;
            in >> camera_orientation.z;
            in >> camera_orientation.w;

            camera_orientation.normalise();

            zeroNode->setPosition(camera_zero);
            mCamera->setPosition(camera_position);
            mCamera->setOrientation(camera_orientation);
            update();
            in.close();
            return;
        }
    }

    //if no view file was present, we set up automatically the view.
    showEntireScene();
    update();
}



//*****************************************************************************************
// Resize
void QtOgreViewer::setSizeW( int size )
{
    update();
    if (mRenderWindow != NULL)
    {
        mRenderWindow->resize( size, mRenderWindow->getHeight());
        mRenderWindow->windowMovedOrResized();
    }
    QWidget::resize(size,this->height());
}

void QtOgreViewer::setSizeH( int size )
{
    update();
    if (mRenderWindow != NULL)
    {
        mRenderWindow->resize( mRenderWindow->getWidth(), size);
        mRenderWindow->windowMovedOrResized();
    }
    QWidget::resize(this->width(),size);
}


void QtOgreViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+"."+sofa::gui::SofaGUI::GetGUIName()+".view";
        std::ofstream out(viewFileName.c_str());
        if (!out.fail())
        {

            Ogre::Vector3 position_cam = mCamera->getPosition();
            Ogre::Vector3 position_zero = zeroNode->getPosition();
            Ogre::Quaternion orientation_cam = mCamera->getOrientation();

            out << position_zero[0] << " " << position_zero[1] << " " << position_zero[2] << "\n";
            out << position_cam[0] << " " << position_cam[1] << " " << position_cam[2] << "\n";
            out << orientation_cam.x << " " << orientation_cam.y << " " << orientation_cam.z << " " << orientation_cam.w << "\n";

            out.close();
        }
        std::cout << "View parameters saved in "<<viewFileName<<std::endl;
    }
}


} //qtogre
} //viewer
} //qt
} //gui
} //sofa
