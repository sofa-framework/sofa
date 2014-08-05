import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0

Rectangle {
    Layout.fillWidth: true
    height: 30
    MenuBar {
        Menu {
            title: "&File"
            visible: true

            MenuItem { action: openAction }
            MenuItem { action: reloadAction }
            MenuItem { action: saveAction }
            MenuItem { action: saveAsAction }
            MenuSeparator { }
            MenuItem {
                text: "Exit"
                shortcut: "Ctrl+Q"
                onTriggered: close()
            }
        }
        Menu {
            title: "&Edit"
            //MenuItem { action: cutAction }
            //MenuItem { action: copyAction }
            //MenuItem { action: pasteAction }
            MenuSeparator { }
            MenuItem {
                text: "Empty"
                enabled: false
            }
        }
        Menu {
            title: "&View"
            MenuItem {
                text: "Empty"
                enabled: false
            }
        }
        Menu {
            title: "&Help"
            MenuItem {
                text: "Empty"
                enabled: false
            }
        }
    }
}
