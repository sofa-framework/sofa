/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
* Contributors:
*   - damien.marchal@univ-lille1.fr
*******************************************************************************/

#include <QWidget>
#include "../SofaGUIQt.h"

////////////////////////////// FORWARD DECLARATION ////////////////////////////
class QLineEdit ;
class QTextBrowser ;

namespace sofa {
    namespace gui {
        namespace qt {
            class RealGUI ;
            class BrowserHistory ;
}}}


/////////////////////////////// DECLARATION //////////////////////////////////
namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_SOFAGUIQT_API DocBrowser : public QWidget
{
Q_OBJECT

public:
    DocBrowser(RealGUI* g) ;
    ~DocBrowser() ;

    void loadHtml(const std::string& filename) ;
    virtual void showEvent(QShowEvent*) override ;

public slots:
    void goToPrev() ;
    void goTo(const QUrl& u) ;
    void goToHome() ;
    void flipVisibility() ;

signals:
    void visibilityChanged(bool) ;

private:
    /// This is use to retrieve the execution context when browsing the
    /// history of sofa scene execution.
    BrowserHistory* m_browserhistory ;

    QLineEdit*      m_lineEdit;
    QTextBrowser*   m_htmlPage;
    RealGUI*        m_realgui ;
} ;

}
}
}
