import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

MenuBar {
    Menu {
        title: "&File"
        visible: true

        MenuItem {action: openAction}
        MenuItem {action: reloadAction}
        //MenuItem {action: saveAction}
        //MenuItem {action: saveAsAction}
        MenuSeparator {}
        MenuItem {action: exitAction}
    }
    Menu {
        title: "&Edit"
        //MenuItem {action: cutAction}
        //MenuItem {action: copyAction}
        //MenuItem {action: pasteAction}
        //MenuSeparator {}
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
        MenuItem {action: aboutAction}
    }
}
