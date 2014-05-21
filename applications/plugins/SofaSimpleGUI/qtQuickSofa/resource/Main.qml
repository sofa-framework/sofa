import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0

import Window 1.0
import "gui"

Window {
    id: window

    title: "QtQuick SofaViewer"
    width: 1280
    height: 720    
    property string filePath: ""

	// sofa scene
	Scene {
		id: scene
	}

    // dialog
    FileDialog {
        id: openDialog
        nameFilters: ["Scene files (*.xml *.scn *.pscn *.py *.simu *)"]
        onAccepted: {
            filePath = fileUrl
            Sofa.open(filePath)
        }
    }

    FileDialog {
        id: saveDialog
        selectExisting: false
        nameFilters: ["Scene files (*.scn)"]
        onAccepted: {
            filePath = fileUrl
            Sofa.save(filePath)
        }
    }

    // action

    Action {
        id: openAction
        text: "&Open..."
        shortcut: "Ctrl+O"
        onTriggered: openDialog.open()
        tooltip: "Open a Sofa Scene"
    }

    Action {
        id: reloadAction
        text: "&Reload"
        shortcut: "Ctrl+R"
        onTriggered: Sofa.reload()
        tooltip: "Reload the Sofa Scene"
    }

    Action {
        id: saveAction
        text: "&Save"
        shortcut: "Ctrl+S"
        onTriggered: if(0 == filePath.length) saveDialog.open(); else Sofa.save(filePath)
        tooltip: "Save the Sofa Scene"
    }

    Action {
        id: saveAsAction
        text: "&Save As..."
        onTriggered: saveDialog.open()
        tooltip: "Save the Sofa Scene at a specific location"
    }

    Action {
        id: saveScreenshot
        text: "Save screenshot"
        shortcut: "Ctrl+P"
        onTriggered: window.saveScreenshot()
        tooltip: "Take a screenshot of the window and save it next to the application executable"
    }

    // content

    ColumnLayout {
        id: mainLayout
        anchors.fill: parent

        Header {
            id: header
            Layout.fillWidth: true
            height: 30
        }

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
