import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import QtQuick.Controls.Private 1.0
import Qt.labs.settings 1.0
import Window 1.0

Window {
    id: root
    readonly property alias window: root

    title: Qt.application.name

    // sofa scene
    property alias scene: scene
    Scene {
        id: scene
    }

    overrideCursorShape: 0

    property string statusMessage
    property int    statusDuration: 5000

    onStatusMessageChanged: clearStatusTimer.restart()

    Timer {
        id: clearStatusTimer
        running: false
        repeat: false
        interval: statusDuration
        onTriggered: statusMessage = ""
    }

    //////////////////////////////////////////////////

    // ui settings
    property alias globalUiSettings: globalUiSettings
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

            application.clearSettingGroup("ui_" + uiId.toString());
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
    property alias recentSettings: recentSettings
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

    //////////////////////////////////////////////////

    property MenuBar menuBar: null
    property Item toolBar
    property Item statusBar

    property alias contentItem : contentArea

    /*! \internal */
    property real __topBottomMargins: contentArea.y + statusBarArea.height
    /*! \internal */
    readonly property real __qwindowsize_max: (1 << 24) - 1

    /*! \internal */
    property real __width: 0
    Binding {
        target: root
        property: "__width"
        when: root.minimumWidth <= root.maximumWidth
        value: Math.max(Math.min(root.maximumWidth, contentArea.implicitWidth), root.minimumWidth)
    }
    /*! \internal */
    property real __height: 0
    Binding {
        target: root
        property: "__height"
        when: root.minimumHeight <= root.maximumHeight
        value: Math.max(Math.min(root.maximumHeight, contentArea.implicitHeight), root.minimumHeight)
    }
    width: contentArea.__noImplicitWidthGiven ? 0 : __width
    height: contentArea.__noImplicitHeightGiven ? 0 : __height

    minimumWidth: contentArea.__noMinimumWidthGiven ? 0 : contentArea.minimumWidth
    minimumHeight: contentArea.__noMinimumHeightGiven ? 0 : (contentArea.minimumHeight + __topBottomMargins)

    maximumWidth: Math.min(__qwindowsize_max, contentArea.maximumWidth)
    maximumHeight: Math.min(__qwindowsize_max, contentArea.maximumHeight + __topBottomMargins)
    onToolBarChanged: { if (toolBar) { toolBar.parent = toolBarArea } }

    onStatusBarChanged: { if (statusBar) { statusBar.parent = statusBarArea } }

    //onVisibleChanged: { if (visible && menuBar) { menuBar.__parentWindow = root } }

    /*! \internal */
    default property alias data: contentArea.data

    color: syspal.window

    flags: Qt.Window | Qt.WindowFullscreenButtonHint |
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint |
        Qt.WindowCloseButtonHint | Qt.WindowFullscreenButtonHint

    SystemPalette {id: syspal}

    Item {
        id: backgroundItem
        anchors.fill: parent

        Keys.forwardTo: menuBar ? [menuBar.__contentItem] : []

        ContentItem {
            id: contentArea
            anchors.top: toolBarArea.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: statusBarArea.top
        }

        Item {
            id: toolBarArea
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            implicitHeight: childrenRect.height
            height: visibleChildren.length > 0 ? implicitHeight: 0
        }

        Item {
            id: statusBarArea
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            implicitHeight: childrenRect.height
            height: visibleChildren.length > 0 ? implicitHeight: 0
        }

        //onVisibleChanged: if (visible && menuBar) menuBar.__parentWindow = root

        states: State {
            name: "hasMenuBar"
            when: menuBar && !menuBar.__isNative

            ParentChange {
                target: menuBar.__contentItem
                parent: backgroundItem
            }

            PropertyChanges {
                target: menuBar.__contentItem
                x: 0
                y: 0
                width: backgroundItem.width
            }

            AnchorChanges {
                target: toolBarArea
                anchors.top: menuBar.__contentItem.bottom
            }
        }
    }
}
