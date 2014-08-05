import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0

import "gui"

ApplicationWindow {
    id: window

    title: "QtQuick SofaViewer"
    width: 1280
    height: 720    
    property string filePath: ""

    menuBar: MenuBar {
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

	// sofa scene
    Scene {
		id: scene
    }

    // dialog
    FileDialog {
        id: openDialog
        nameFilters: ["Scene files (*.xml *.scn *.pscn *.py *.simu *)"]
        onAccepted: {
            filePath = fileUrl;
            scene.open(fileUrl);
        }
    }

    FileDialog {
        id: saveDialog
        selectExisting: false
        nameFilters: ["Scene files (*.scn)"]
        onAccepted: {
            filePath = fileUrl;
            scene.save(filePath);
        }
    }

    // action

    Action {
        id: openAction
        text: "&Open..."
        shortcut: "Ctrl+O"
        onTriggered: openDialog.open();
        tooltip: "Open a Sofa Scene"
    }

    Action {
        id: reloadAction
        text: "&Reload"
        shortcut: "Ctrl+R"
        onTriggered: scene.reload();
        tooltip: "Reload the Sofa Scene"
    }

    Action {
        id: saveAction
        text: "&Save"
        shortcut: "Ctrl+S"
        onTriggered: if(0 == filePath.length) saveDialog.open(); else scene.save(filePath);
        tooltip: "Save the Sofa Scene"
    }

    Action {
        id: saveAsAction
        text: "&Save As..."
        onTriggered: saveDialog.open();
        tooltip: "Save the Sofa Scene at a specific location"
    }

    // content

    ColumnLayout {
        id: mainLayout
        anchors.fill: parent

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true

            SceneGraph {
                id: sceneGraph

                Layout.fillWidth: true
                Layout.fillHeight: true
                width: 30
            }

            Viewer {
                id: viewer

                Layout.fillWidth: true
                Layout.fillHeight: true
                width: 70

                scene: scene
            }
        }

        Footer {
            id: footer

            Layout.fillWidth: true
            height: 30
        }
    }
}
