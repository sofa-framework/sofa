/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef ADDPRESET_H
#define ADDPRESET_H

#include <sofa/simulation/Node.h>

#include <QDialog>

class QLabel;
class QLineEdit;
class QPushButton;

namespace sofa
{

namespace gui
{

namespace qt
{
using sofa::simulation::Node;

class AddPreset: public QDialog
{
    Q_OBJECT;
public:

    AddPreset(QWidget* parent);
    void setElementPresent(bool *elementPresent);
    void setParentNode(Node* parentNode) {node=parentNode;}
    void setPresetFile(std::string p) {presetFile=p;}
    void setPath(std::string p) {fileName=p;}
    void setRelativePath(std::string p) {relative=p;}
    void clear();

public slots:
    void fileOpen();
    void accept();

signals:
    void loadPreset(Node*,std::string,std::string*, std::string,std::string,std::string);

protected:
    std::string fileName;
    std::string relative;
    std::string presetFile;
    Node *node;

private:
    QLabel *openFileText0;
    QLineEdit *openFilePath0;
    QPushButton *openFileButton0;
    QLabel *openFileText1;
    QLineEdit *openFilePath1;
    QPushButton *openFileButton1;
    QLabel *openFileText2;
    QLineEdit *openFilePath2;
    QPushButton *openFileButton2;
    QLineEdit *positionX;
    QLineEdit *positionY;
    QLineEdit *positionZ;
    QLineEdit *rotationX;
    QLineEdit *rotationY;
    QLineEdit *rotationZ;
    QLineEdit *scaleX;
    QLineEdit *scaleY;
    QLineEdit *scaleZ;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif
