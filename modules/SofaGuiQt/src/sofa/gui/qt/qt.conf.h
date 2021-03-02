/****************************************************************************
** Resource object code
**
** Created by: The Resource Compiler for Qt version 5.15.2
*****************************************************************************/

#include <sofa/config.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

static std::vector<unsigned char> qt_resource_data = {};

static const unsigned char qt_resource_name[] = {
  // qt
  0x0,0x2,
  0x0,0x0,0x7,0x84,
  0x0,0x71,
  0x0,0x74,
  // etc
  0x0,0x3,
  0x0,0x0,0x6c,0xa3,
  0x0,0x65,
  0x0,0x74,0x0,0x63,
  // qt.conf
  0x0,0x7,
  0x8,0x74,0xa6,0xa6,
  0x0,0x71,
  0x0,0x74,0x0,0x2e,0x0,0x63,0x0,0x6f,0x0,0x6e,0x0,0x66,
};

static const unsigned char qt_resource_struct[] = {
  // :
  0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x1,
  // :/qt
  0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x2,
  // :/qt/etc
  0x0,0x0,0x0,0xa,0x0,0x2,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x3,
  // :/qt/etc/qt.conf
  0x0,0x0,0x0,0x16,0x0,0x0,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x0,
};

#ifdef QT_NAMESPACE
#  define QT_RCC_PREPEND_NAMESPACE(name) ::QT_NAMESPACE::name
#  define QT_RCC_MANGLE_NAMESPACE0(x) x
#  define QT_RCC_MANGLE_NAMESPACE1(a, b) a##_##b
#  define QT_RCC_MANGLE_NAMESPACE2(a, b) QT_RCC_MANGLE_NAMESPACE1(a,b)
#  define QT_RCC_MANGLE_NAMESPACE(name) QT_RCC_MANGLE_NAMESPACE2( \
        QT_RCC_MANGLE_NAMESPACE0(name), QT_RCC_MANGLE_NAMESPACE0(QT_NAMESPACE))
#else
#   define QT_RCC_PREPEND_NAMESPACE(name) name
#   define QT_RCC_MANGLE_NAMESPACE(name) name
#endif

#ifdef QT_NAMESPACE
namespace QT_NAMESPACE {
#endif

bool qRegisterResourceData(int, const unsigned char *, const unsigned char *, const unsigned char *);
bool qUnregisterResourceData(int, const unsigned char *, const unsigned char *, const unsigned char *);

#ifdef QT_NAMESPACE
}
#endif


namespace sofa::gui::qt {
bool loadQtConfWithCustomPrefix(const std::string& qtConfPath, const std::string& prefix)
{
    if( ! qt_resource_data.empty() )
    {
        msg_warning("qt.conf.h") << "loadQtConfWithCustomPrefix can only be called once.";
        return false;
    }

    // Qt wants paths with slashes
    const std::string qtConfPathClean = FileSystem::cleanPath(qtConfPath, FileSystem::SLASH);
    const std::string prefixClean = FileSystem::cleanPath(prefix, FileSystem::SLASH);

    if ( ! FileSystem::isDirectory(prefixClean) )
    {
        msg_warning("qt.conf.h") << "Directory not found " << prefixClean;
        return false;
    }

    std::ifstream inputFile(qtConfPathClean);
    if ( ! inputFile.is_open() )
    {
        msg_warning("qt.conf.h") << "Cannot open file " << qtConfPathClean;
        return false;
    }

    std::stringstream output;
    std::string inputLine;
    while ( std::getline(inputFile, inputLine) )
    {
        if ( inputLine.find("Prefix") != std::string::npos )
        {
            output << "  Prefix = " << prefixClean;
        }
        else
        {
            output << inputLine;
        }
#if defined(WIN32)
        output << '\r' << '\n';
#elif defined(__APPLE__)
        output << '\r';
#else
        output << '\n';
#endif
    }

    std::vector<char> data = std::vector<char>(std::istreambuf_iterator<char>(output), std::istreambuf_iterator<char>());
    int dataSize = data.size();
    qt_resource_data.resize(4 + dataSize);

    for ( int i = 0 ; i < 4 + dataSize ; i++ )
    {
        if ( i < 4 ) // first 4 bytes are for size
        {
            qt_resource_data[3 - i] = static_cast<unsigned char>( (dataSize >> (i * 8)) & 0xFF );
        }
        else // next bytes are for data
        {
            qt_resource_data[i] = static_cast<unsigned char>( data[i - 4] );
        }
    }

    return QT_RCC_PREPEND_NAMESPACE(qRegisterResourceData)(1, qt_resource_struct, qt_resource_name, &qt_resource_data[0]);
}
} // namespace sofa::gui::qt
