import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import QtQuick.Controls.Styles 1.3
import SofaBasics 1.0
import SofaTools 1.0
import SofaWidgets 1.0
import "qrc:/SofaCommon/SofaSettingsScript.js" as SofaSettingsScript

ApplicationWindow {
    id: root
    width: 1280
    height: 720
    title: Qt.application.name

    style: ApplicationWindowStyle {
        background: null
    }

    Component.onCompleted: {
        if (Qt.application.arguments.length>1)
            scene.source = "file:" + Qt.application.arguments[1]
        else {
            if(0 !== SofaSettingsScript.Recent.scenes.length)
                scene.source = SofaSettingsScript.Recent.mostRecent();
            else
                scene.source = "file:Demos/caduceus.scn";
        }

        visible = true;
    }

    // dialog
    property FileDialog openSofaSceneDialog: openSofaSceneDialog
    FileDialog {
        id: openSofaSceneDialog
        nameFilters: ["Scene files (*.xml *.scn *.pscn *.py *.simu *)"]
        onAccepted: {
            scene.source = fileUrl;
        }
    }

    property FileDialog saveSofaSceneDialog: saveSofaSceneDialog
    FileDialog {
        id: saveSofaSceneDialog
        selectExisting: false
        nameFilters: ["Scene files (*.scn)"]
        onAccepted: {
            scene.save(fileUrl);
        }
    }

    menuBar: DefaultMenuBar {
        id: menuBar
        scene: scene
    }

    Scene {
        id: scene

        property var listModel: SceneListModel {id : listModel}

        onStepEnd: listModel.update()
    }

    DynamicSplitView {
        id: dynamicSplitView
        anchors.fill: parent
        uiId: 1
        sourceComponent: Component {
            DynamicContent {
                defaultContentName: "Viewer"
                sourceDir: "qrc:/SofaWidgets"
                properties: {"scene": scene}
            }
        }
    }

    statusBar: DefaultStatusBar {
        id: statusBar
        scene: scene
    }
}
