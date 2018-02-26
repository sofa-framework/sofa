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
#include <QDesktopServices>
#include <QTextBrowser>
#include <QLineEdit>

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include "QDocBrowser.h"
#include "../RealGUI.h"
#include <sofa/gui/GuiDataRepository.h>

namespace sofa
{
namespace gui
{
namespace qt
{

///////////////////////////// PRIVATE OBJECTS //////////////////////////////////
/// May be moved to their own .cpp/.hh if one day someone needs them.


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

const char* asCStr(const QString& s)
{
    return s.toStdString().c_str() ;
}

DocBrowser::~DocBrowser()
{

}

DocBrowser::DocBrowser(RealGUI* g) : QWidget()
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
    prev->setIcon(QIcon(asQStr( GuiDataRepository.getFile("icons/back.png")))) ;
    connect(prev, SIGNAL(clicked()), this, SLOT(goToPrev()));
    bgl->addWidget(prev);

    /// Here we do an autocompletion and search for components.
    m_lineEdit = new QLineEdit();
    bgl->addWidget(m_lineEdit);

    QPushButton* home=new QPushButton();
    home->setIcon(QIcon(asQStr( GuiDataRepository.getFile("icons/home.png"))));
    connect(home, SIGNAL(clicked()), this, SLOT(goToHome()));
    bgl->addWidget(home) ;

    /// Add the html browser to visualize the documentation
    m_htmlPage = new QTextBrowser(this);
    verticalLayout->addWidget(m_htmlPage);

    /// We want click on internal links (file://) to be routed to the the goTo function to
    /// load the sofa file.
    m_htmlPage->setOpenLinks(false) ;
    connect(m_htmlPage, SIGNAL(anchorClicked(const QUrl&)),
            this, SLOT(goTo(const QUrl&)));
}

void DocBrowser::loadHtml(const std::string& filename)
{
    bool showView = true ;
    std::string htmlfile = filename ;
    std::string rootdir = FileSystem::getParentDirectory(filename) ;

    std::string extension=FileSystem::getExtension(filename);
    htmlfile.resize(htmlfile.size()-extension.size()-1);
    htmlfile+=".html";

    /// Check if there exists an .html  file associated with the provided file.
    /// If nor and the history is empty we load a default document from the share repository.
    if (!DataRepository.findFile(htmlfile, "", NULL))
    {
        if( m_browserhistory->size() == 0 )
        {
            htmlfile = GuiDataRepository.getFile("docs/runsofa.html").c_str() ;
        }
        showView = false ;
    }

    /// Check if the page we want to load is already loaded on the top of the history
    /// If this is the case there is no need to reload it.
    if(m_browserhistory->size() != 0)
    {
        if(m_browserhistory->current().m_htmlfile == htmlfile )
            return ;
    }

    /// Check if either the scene specific html or default provided can be loaded.
    /// If so...load the page and add the entry into the history.
    if (DataRepository.findFile(htmlfile, "", NULL))
    {
        m_htmlPage->setSearchPaths(QStringList(QString(rootdir.c_str())));
        m_htmlPage->setSource(QUrl::fromLocalFile(QString(htmlfile.c_str())) );
        m_browserhistory->push(htmlfile, filename, rootdir) ;
    }

    setVisible(showView);
}

void DocBrowser::goToPrev()
{
    /// Drop the old entry.
    BrowserHistoryEntry entry = m_browserhistory->pop() ;

    /// Get the previous one
    entry = m_browserhistory->current() ;
    m_htmlPage->setSearchPaths({asQStr(entry.m_rootdir)});
    m_htmlPage->setSource(asQStr(entry.m_htmlfile)) ;
}

void DocBrowser::goTo(const QUrl& u)
{
    BrowserHistoryEntry entry = m_browserhistory->current() ;
    std::string path = FileSystem::cleanPath(entry.m_rootdir + "/" + u.path().toStdString()) ;
    std::string extension=FileSystem::getExtension(path);

    if(path.empty())
    {
        return ;
    }

    /// Check if the path is pointing to an html file.
    if ( extension == "html" )
    {
        loadHtml(path.c_str()) ;
        return ;
    }

    /// Check if the path is pointing to a sofa scene. If so
    /// open the scene
    const auto exts = SceneLoaderFactory::getInstance()->extensions() ;
    if ( std::find(exts.begin(), exts.end(), extension) != exts.end() )
    {
        m_realgui->fileOpen(path) ;
        return ;
    }

    QDesktopServices::openUrl(u) ;
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

} ///namespace qt
} ///namespace gui
} ///namespace sofa
