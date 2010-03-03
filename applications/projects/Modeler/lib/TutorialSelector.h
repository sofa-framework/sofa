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
#ifndef SOFA_TUTORIALSELECTOR_H
#define SOFA_TUTORIALSELECTOR_H

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <QKeyEvent>
#else
#include <qlistview.h>
#include <qevent.h>
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

//Tinyxml library
#include <tinyxml.h>
#include <tinystr.h>

#include <map>

namespace sofa
{

namespace gui
{

namespace qt
{



class TutorialSelector : public Q3ListView
{
    struct Tutorial
    {
        Tutorial() {};
        Tutorial(const std::string &n, const std::string &scene, const std::string &html)
            :name(n), sceneFilename(scene), htmlFilename(html) {};
        std::string name;
        std::string sceneFilename;
        std::string htmlFilename;
    };

    Q_OBJECT
public:
    TutorialSelector(const std::string &fileTutorials, QWidget* parent = 0);

    void keyPressEvent ( QKeyEvent * e );
    void usingScene(const std::string &filename);
public  slots:
#ifdef SOFA_QT4
    void openTutorial( Q3ListViewItem * );
#else
    void openTutorial( QListViewItem * );
#endif
signals:
    void openTutorial(const std::string &filename);
    void openHTML(const std::string &filename);

protected:
    void init(const std::string &fileTutorials);
    void openNode(TiXmlNode* node, Q3ListViewItem *parent=NULL, bool isRoot=false);
    void openAttribute(TiXmlElement* element,  Q3ListViewItem *item);

    std::map< Q3ListViewItem *, Tutorial> itemToTutorial;
};

}
}
}

#endif
