import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import "qrc:/SofaCommon/SofaSettingsScript.js" as SofaSettingsScript

MenuBar {
    id: menuBar

    property Scene scene

    property list<QtObject> objects: [

        // action
        Action {
            id: openAction
            text: "&Open..."
            shortcut: "Ctrl+O"
            onTriggered: openSofaSceneDialog.open();
            tooltip: "Open a Sofa Scene"
        },

        Action {
            id: openRecentAction
            onTriggered: {
                var title = source.text.toString();
                var source = title.replace(/^.*"(.*)"$/m, "$1");
                scene.source = "file:" + source
            }
        },

        Action {
            id: clearRecentAction
            text: "&Clear"
            onTriggered: SofaSettingsScript.Recent.clear();
            tooltip: "Clear history"
        },

        Action {
            id: reloadAction
            text: "&Reload"
            shortcut: "Ctrl+R"
            onTriggered: scene.reload();
            tooltip: "Reload the Sofa Scene"
        },

//        Action {
//            id: saveAction
//            text: "&Save"
//            shortcut: "Ctrl+S"
//            onTriggered: if(0 == filePath.length) saveSofaSceneDialog.open(); else scene.save(filePath);
//            tooltip: "Save the Sofa Scene"
//        },

        Action {
            id: saveAsAction
            text: "&Save As..."
            onTriggered: saveSofaSceneDialog.open();
            tooltip: "Save the Sofa Scene at a specific location"
        },

        Action
        {
            id: exitAction
            text: "&Exit"
            shortcut: "Ctrl+Q"
            onTriggered: close()
        },

        MessageDialog {
            id: aboutDialog
            title: "About"
            text: "Welcome in the " + Qt.application.name +" Application"
            onAccepted: visible = false
        },

        Action
        {
            id: aboutAction
            text: "&About"
            onTriggered: aboutDialog.visible = true;
            tooltip: "What is this application ?"
        }
    ]

    Menu {
        title: "&File"
        visible: true

        MenuItem {action: openAction}
        Menu {
            id: recentMenu
            title: "Recent scenes"

            visible: 0 !== items.length

            function update() {
                recentMenu.clear();
                var sceneList = SofaSettingsScript.Recent.sceneList();
                if(0 === sceneList.length)
                    return;

                for(var j = 0; j < sceneList.length; ++j) {
                    var sceneSource = sceneList[j];
                    if(0 === sceneSource.length)
                        continue;

                    var sceneName = sceneSource.replace(/^.*[//\\]/m, "");
                    var title = j.toString() + " - " + sceneName + " - \"" + sceneSource + "\"";

                    var openRecentItem = recentMenu.addItem(title);
                    openRecentItem.action = openRecentAction;

                    if(10 === recentMenu.items.length)
                        break;
                }

                if(0 === recentMenu.items.length)
                    return;

                recentMenu.addSeparator();
                var clearRecentItem = recentMenu.addItem("Clear");
                clearRecentItem.action = clearRecentAction;
            }

            Component.onCompleted: recentMenu.update()

            Connections {
                target: SofaSettingsScript.Recent
                onScenesChanged: recentMenu.update()
            }
        }

        MenuItem {action: reloadAction}
        //MenuItem {action: saveAction}
        //MenuItem {action: saveAsAction}
        MenuSeparator {}
        MenuItem {action: exitAction}
    }
/*
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
*/
    Menu {
        title: "&Help"
        MenuItem {action: aboutAction}
    }
}
