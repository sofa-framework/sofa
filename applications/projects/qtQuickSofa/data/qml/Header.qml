import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import Qt.labs.settings 1.0

MenuBar {
    Menu {
        title: "&File"
        visible: true

        MenuItem {action: openAction}
        Menu {
            id: recentMenu
            title: "Recent scenes"

            visible: 0 !== items.length

            Connections {
                target: recentSettings
                onScenesChanged: {
                    recentMenu.clear();
                    var sceneList = recentSettings.scenes.split(';');
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
            }
        }

        MenuItem {action: reloadAction}
        MenuItem {action: resetAction}
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
