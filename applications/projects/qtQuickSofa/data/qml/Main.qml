import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import SofaBasics 1.0
import SofaTools 1.0
import SofaWidgets 1.0
import Qt.labs.settings 1.0

Window {
    id: window

    title: "qtQuickSofa"
    width: 1280
    height: 720
    property string filePath: ""

    overrideCursorShape: 0

    // sofa scene
    Scene {
        id: scene
    }

    // ui settings
    Settings {
        id: globalUiSettings
        category: "ui"

        property string uiIds: ";"

        function generateUiId() {
            var uiId = 1;
            var uiIdList = uiIds.split(";");

            var previousId = 0;
            for(var i = 0; i < uiIdList.length; ++i) {
                if(0 === uiIdList[i].length)
                    continue;

                uiId = Number(uiIdList[i]);

                if(previousId + 1 !== uiId) {
                    uiId = previousId + 1;
                    break;
                }

                previousId = uiId;
                ++uiId;
            }

            return uiId;
        }

        function addUiId(uiId) {
            if(0 === uiId)
                return;

            if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
                uiIds += uiId.toString() + ";";
                update();
            }
        }

        function removeUiId(uiId) {
            if(0 === uiId)
                return;

            uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

            clearSettingGroup("ui_" + uiId.toString());
        }

        function replaceUiId(previousUiId, uiId) {
            if(0 === uiId)
                return;

            uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

            if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
                uiIds += uiId.toString() + ";";
                update();
            }
        }

        function update() {
            var uiIdList = uiIds.split(";");
            uiIdList.sort(function(a, b) {return Number(a) - Number(b);});

            uiIds = ";";
            for(var i = 0; i < uiIdList.length; ++i)
                if(0 !== uiIdList[i].length)
                    uiIds += uiIdList[i] + ";";
        }
    }

    // recent opened scene settings
    Settings {
        id: recentSettings
        category: "recent"

        Component.onCompleted: {
            scenesChanged();
        }

        property string scenes

        function add(sceneSource) {
            sceneSource += ";";
            scenes = scenes.replace(sceneSource, "");
            scenes = sceneSource + scenes;
        }

        function clear() {
            scenes = "";
        }
    }

    // dialog
    FileDialog {
        id: openDialog
        nameFilters: ["Scene files (*.xml *.scn *.pscn *.py *.simu *)"]
        onAccepted: {
            scene.source = fileUrl;
        }
    }

    FileDialog {
        id: saveDialog
        selectExisting: false
        nameFilters: ["Scene files (*.scn)"]
        onAccepted: {
            scene.save(fileUrl);
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
        id: openRecentAction
        onTriggered: {
            var title = source.text.toString();
            var source = title.replace(/^.*"(.*)"$/m, "$1");
            scene.source = "file:" + source
        }
    }

    Action {
        id: clearRecentAction
        text: "&Clear"
        onTriggered: recentSettings.clear();
        tooltip: "Clear history"
    }

    Action {
        id: reloadAction
        text: "&Reload"
        shortcut: "Ctrl+R"
        onTriggered: scene.reload();
        tooltip: "Reload the Sofa Scene"
    }

    Action {
        id: resetAction
        text: "&Reset"
        shortcut: "Ctrl+Alt+R"
        onTriggered: scene.reset();
        tooltip: "Reset the simulation"
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

    Action
    {
        id: exitAction
        text: "&Exit"
        shortcut: "Ctrl+Q"
        onTriggered: close()
    }

    MessageDialog {
        id: aboutDialog
        title: "About"
        text: "Welcome in the " + window.title +" Application"
        onAccepted: visible = false
    }

    Action
    {
        id: aboutAction
        text: "&About"
        onTriggered: aboutDialog.visible = true;
        tooltip: "What is this application ?"
    }

// header

    menuBar: Header {}

// content

    Item {
        anchors.fill: parent

        RowLayout {
            anchors.fill: parent
            spacing: 0

            Component {
                id: dynamicContentComponent

                DynamicContent {
                    id: dynamicContent
                    defaultContentName: "Viewer"
                    sourceDir: "qrc:///data/qml/component/SofaWidgets"

                    Binding {
                        target: dynamicContent.contentItem
                        property: "scene"
                        value: scene
                        when: dynamicContent.contentItem ? true : false
                    }
                }
            }

            DynamicSplitView {
                id: dynamicSplitView
                uiId: 1
                sourceComponent: dynamicContentComponent
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }

// footer

    property alias statusMessage: footer.statusMessage
    statusBar: Footer {
        id: footer
    }
}
