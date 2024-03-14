/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <QDesktopServices>
#include <QTextBrowser>
#include <QLineEdit>
#include <QFileInfo>
#include <QUrlQuery>
#include <QDir>

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include "QDocBrowser.h"
#include "../RealGUI.h"
#include <sofa/gui/common/GuiDataRepository.h>

#include <iostream>

using namespace sofa::gui::common;

namespace sofa::gui::qt 
{

///////////////////////////// PRIVATE OBJECTS //////////////////////////////////
/// May be moved to their own .cpp/.hh if one day someone needs them.

SofaEnrichedPage::SofaEnrichedPage(QObject* parent) : QWebEnginePage(parent)
{
}

bool SofaEnrichedPage::isSofaTarget(const QUrl& u)
{
    if( u.fileName() == QString("sofa") && u.hasQuery() )
    {
        return true ;
    }else if( u.isLocalFile() && ! u.hasQuery() )
    {
        return true ;
    }
    return false;
}

bool SofaEnrichedPage::acceptNavigationRequest(const QUrl & url,
                                               QWebEnginePage::NavigationType type,
                                               bool )
{
    if (type == QWebEnginePage::NavigationTypeLinkClicked)
    {
        if( isSofaTarget(url) )
        {
            emit linkClicked(url);
            return false;
        }
    }
    return true;
}



//////////////////////////////// BrowserHistory ////////////////////////////////
/// Hold an history entry which include the .html file, the sofa scene file and the
/// root directory. This was needed to implement the backward function of the
/// doc browser as the internal one implemented in QTextBrowser was only capable
/// of storing the .html file.
class BrowserHistoryEntry
{
public:
    std::string m_htmlfile ;
    std::string m_scenefile ;
    std::string m_rootdir ;

    BrowserHistoryEntry(const std::string& html,
                        const std::string& scene,
                        const std::string& rootdir)
    {
        m_htmlfile = html ;
        m_scenefile = scene ;
        m_rootdir = rootdir ;
    }
};

class BrowserHistory
{
public:
    std::vector<BrowserHistoryEntry> m_history ;
    void push(const std::string& html, const std::string& scene, const std::string& rootdir) ;
    BrowserHistoryEntry current() ;
    BrowserHistoryEntry pop() ;
    int size() ;
};

int BrowserHistory::size()
{
    return m_history.size() ;
}

BrowserHistoryEntry BrowserHistory::current(){
    if(m_history.size()==0)
        throw std::exception() ;

    return *(m_history.end()-1) ;
}

void BrowserHistory::push(const std::string& html, const std::string& scene, const std::string& rootdir)
{
    m_history.push_back(BrowserHistoryEntry(html, scene, rootdir));
}

BrowserHistoryEntry BrowserHistory::pop()
{
    if(m_history.size()==0)
        throw std::exception() ;

    BrowserHistoryEntry entry = *(m_history.end()-1) ;

    if(m_history.size()>1)
        m_history.pop_back() ;

    return entry ;
}
////////////////////////////////////////////////////////////////////////////////


QString asQStr(const std::string& c)
{
    return QString(c.c_str()) ;
}

std::string asStr(const QString& s)
{
    return s.toStdString() ;
}


DocBrowser::~DocBrowser()
{

}

DocBrowser::DocBrowser(RealGUI* g) : QDialog(g)
{
    m_realgui = g ;
    m_browserhistory = new BrowserHistory() ;

    /// Create a top level window
    setGeometry(0,0,600,600);
    setWindowTitle("Doc browser");

    /// Create the first button line
    QVBoxLayout *verticalLayout = new QVBoxLayout(this);
    QWidget *bg = new QWidget() ;
    QHBoxLayout *bgl = new QHBoxLayout(bg) ;
    bg->setLayout(bgl);
    verticalLayout->addWidget(bg) ;

    QPushButton* prev=new QPushButton() ;
    prev->setIcon(QIcon(asQStr(common::GuiDataRepository.getFile("icons/back.png")))) ;
    connect(prev, SIGNAL(clicked()), this, SLOT(goToPrev()));
    bgl->addWidget(prev);

    /// Here we do an autocompletion and search for components.
    m_lineEdit = new QLineEdit();
    bgl->addWidget(m_lineEdit);

    QPushButton* home=new QPushButton();
    home->setIcon(QIcon(asQStr(common::GuiDataRepository.getFile("icons/home.png"))));
    connect(home, SIGNAL(clicked()), this, SLOT(goToHome()));
    bgl->addWidget(home) ;

    /// Add the html browser to visualize the documentation
    m_htmlPage = new QWebEngineView(this);
    SofaEnrichedPage* pp = new SofaEnrichedPage();
    m_htmlPage->setPage(pp);

    verticalLayout->addWidget(m_htmlPage, 1);

    /// We want click on internal links (file://) to be routed to the the goTo function to
    /// load the sofa file.
    connect(m_htmlPage, SIGNAL(urlChanged(const QUrl&)), this, SLOT(goTo(const QUrl&)));
    connect(pp, SIGNAL(linkClicked(const QUrl&)), this, SLOT(onLinkClicked(const QUrl&)));
}

void DocBrowser::loadHtml(const std::string& filename)
{
    if (filename.empty())
    {
        return;
    }

    std::string htmlfile = filename ;
    std::string rootdir = FileSystem::getParentDirectory(filename) ;

    const QUrl currenturl = m_htmlPage->page()->url() ;

    if(currenturl.isLocalFile() && currenturl.path() == asQStr(htmlfile))
    {
        return ;
    }

    const std::string extension=FileSystem::getExtension(filename);
    htmlfile.resize(htmlfile.size()-extension.size()-1);
    htmlfile+=".html";

    /// Check if either the scene specific html or default provided can be loaded.
    /// If so...load the page and add the entry into the history.
    if (!DataRepository.findFile(htmlfile, "", NULL))
        return;

    m_htmlPage->load( QUrl::fromLocalFile(QString(htmlfile.c_str())) );

    constexpr bool showView = true ;
    setVisible(showView);
}

void DocBrowser::goToPrev()
{
    m_htmlPage->pageAction(QWebEnginePage::Back)->trigger() ;
}

void DocBrowser::onLinkClicked(const QUrl& u)
{
    msg_info("DocBrowser") << " query to load " << asStr(u.path()) ;
    if( u.fileName() == QString("sofa") && u.hasQuery() )
    {
        m_realgui->playpauseGUI(true) ;
        return ;
    }

    if( u.isLocalFile() && ! u.hasQuery() )
    {
        const QFileInfo theFile(u.toLocalFile());
        const std::string sofafile = asStr( theFile.absoluteDir().absoluteFilePath(u.toLocalFile()) );
        const std::string extension = FileSystem::getExtension(sofafile) ;

        /// Check if the path is pointing to a sofa scene. If so
        /// open the scene
        const auto exts = SceneLoaderFactory::getInstance()->extensions() ;
        if ( std::find(exts.begin(), exts.end(), extension) != exts.end() )
        {
            m_realgui->fileOpen(sofafile, false, false) ;
            return ;
        }
    }
    m_htmlPage->load(u) ;
}

void DocBrowser::goTo(const QUrl& u)
{
    msg_info("DocBrowser") << "Go to " << asStr(u.path()) ;
    if( u.isLocalFile() && u.hasQuery() )
    {
        const QUrlQuery q { u.query() } ;
        if( !q.hasQueryItem("sofafile") ) {
            msg_info("DocBrowser") << "Does not have associated sofa file. " ;
            return ;
        }

        const QFileInfo htmlfile(u.toLocalFile());
        const std::string sofafile = asStr( htmlfile.absoluteDir().absoluteFilePath(q.queryItemValue("sofafile")) );
        const std::string extension = FileSystem::getExtension(sofafile) ;

        /// Check if the path is pointing to a sofa scene. If so
        /// open the scene
        const auto exts = SceneLoaderFactory::getInstance()->extensions() ;
        if ( std::find(exts.begin(), exts.end(), extension) == exts.end() ){
            msg_warning("DocBrowser") << "Unsupported sofa file format. " ;
            return ;
        }
        m_realgui->fileOpen(sofafile, false, false) ;
        return ;
    }
}

void DocBrowser::goToHome()
{
    loadHtml(GuiDataRepository.getFile("docs/runsofa.html").c_str());
}

void DocBrowser::flipVisibility()
{
    if(isVisible())
        hide();
    else
        show();
    emit visibilityChanged(isVisible()) ;
}

void DocBrowser::showEvent(QShowEvent *)
{
    emit visibilityChanged(isVisible()) ;
}

} // namespace sofa::gui::qt 
