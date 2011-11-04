/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

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

#ifndef SOFA_SOFACONFIGURATION_H
#define SOFA_SOFACONFIGURATION_H


#ifdef SOFA_QT4
#include <QMainWindow>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QDir>
#include <QLayout>
#include <QPushButton>
#include <Q3Process>
#else
#include <qmainwindow.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qdir.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <qprocess.h>
typedef QProcess Q3Process;
#endif

#include <set>

#include "ConfigurationParser.h"

namespace sofa
{

namespace gui
{

namespace qt
{


class ConfigWidget: public QWidget
{
    Q_OBJECT
public:
    bool getValue() {return check->isChecked();}

    ConfigWidget(QWidget *parent, DEFINES &d);
    DEFINES &option;

public slots:
    void updateValue(bool);

signals:
    void modified();
protected:
    QHBoxLayout *layout;
    QCheckBox *check;
};

class TextConfigWidget: public ConfigWidget
{
    Q_OBJECT
public:

    TextConfigWidget(QWidget *parent, DEFINES &d);


public slots:
    void updateValue(const QString&);
protected:
    QLineEdit *description;
};


class OptionConfigWidget: public ConfigWidget
{
    Q_OBJECT
public:

    OptionConfigWidget(QWidget *parent, DEFINES &d);
protected:
    QLabel *description;
};


class SofaConfiguration : public QMainWindow
{

    Q_OBJECT
public :

    SofaConfiguration(std::string path_, std::vector< DEFINES >& config);
    ~SofaConfiguration()
    {
    };

    bool getValue(CONDITION &option);

    void processCondition(QWidget *w, CONDITION &c);
    void processDirectory(const QString &dir);
    void processFile(const QFileInfo &info);
public slots:

    void updateOptions();
    void updateConditions();
    void saveConfiguration();

protected slots:
    void redirectStdErr();
    void redirectStdOut();
    void saveConfigurationDone();
protected:
    std::string path;
    std::vector< DEFINES >& data;
    std::vector< ConfigWidget *> options;
    std::set< QWidget *> optionsModified;

    QLineEdit *projectVC;
    QPushButton* saveButton;
    Q3Process* p;
};


}
}
}
#endif
