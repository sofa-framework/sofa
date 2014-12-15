import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.folderlistmodel 2.1
import Qt.labs.settings 1.0

Item {
    id: root
    clip: true

    readonly property bool isDynamicContent: true

    property int uiId: 0
    property int previousUiId: uiId
    onUiIdChanged: {
        globalUiSettings.replaceUiId(previousUiId, uiId);
    }

    Settings {
        id: uiSettings
        category: 0 !== root.uiId ? "ui_" + root.uiId : "dummy"

        property string sourceDir
        property string defaultContentName
        property string currentContentName
        property int    contentUiId
    }

    function init() {
        uiSettings.contentUiId          = Qt.binding(function() {return root.contentUiId;});
        uiSettings.sourceDir            = Qt.binding(function() {return root.sourceDir;});
        uiSettings.defaultContentName   = Qt.binding(function() {return root.defaultContentName;});
        uiSettings.currentContentName   = Qt.binding(function() {return root.currentContentName;});
    }

    function load() {
        if(0 === uiId)
            return;

        root.contentUiId        = uiSettings.contentUiId;
        root.sourceDir          = uiSettings.sourceDir;
        root.defaultContentName = uiSettings.defaultContentName;
        root.currentContentName = uiSettings.currentContentName;
    }

    function setNoSettings() {
        globalUiSettings.removeUiId(uiId);
        uiId = 0;
    }

    property string defaultContentName
    property string currentContentName
    property string sourceDir: "qrc:///data/qml/component/SofaWidgets"
    property int    contentUiId: 0

    onSourceDirChanged: update()
    onCurrentContentNameChanged: update()
    Component.onCompleted: {
        if(0 === root.uiId) {
            if(parent && undefined !== parent.contentUiId && 0 !== parent.contentUiId) {
                root.uiId = parent.contentUiId;
                load();
            }
            else
                root.uiId = globalUiSettings.generateUiId();
        }
        else
            load();

        init();

        update();
    }

    function update() {
        folderListModel.folder = ""; // we must reset the folder property
        folderListModel.folder = sourceDir;
    }

    FolderListModel {
        id: folderListModel
        nameFilters: ["*.qml"]
        showDirs: false
        showFiles: true
        sortField: FolderListModel.Name

        function update() {
            listModel.clear();

            var contentSet = false;

            var previousIndex = comboBox.currentIndex;
            for(var i = 0; i < count; ++i)
            {
                var fileBaseName = get(i, "fileBaseName");
                var filePath = get(i, "filePath").toString();
                if(-1 !== folder.toString().indexOf("qrc:"))
                    filePath = "qrc" + filePath;

                listModel.insert(i, {"fileBaseName": fileBaseName, "filePath": filePath});

                if(0 === root.currentContentName.localeCompare(fileBaseName)) {
                    comboBox.currentIndex = i;
                    contentSet = true;
                }
            }

            if(!contentSet) {
                for(var i = 0; i < count; ++i)
                {
                    var fileBaseName = get(i, "fileBaseName");

                    if(0 === defaultContentName.localeCompare(fileBaseName)) {
                        comboBox.currentIndex = i;
                        break;
                    }
                }
            }

            if(count > 0 && previousIndex === comboBox.currentIndex)
                loaderLocation.refresh();
        }

        onCountChanged: {
            update();
        }
    }

    readonly property alias contentItem: loaderLocation.contentItem
    Item {
        id: loaderLocation
        anchors.fill: parent

        property Item contentItem

        onContentItemChanged: {
            refreshStandbyItem();
        }

        function refresh() {
            if(-1 === comboBox.currentIndex || comboBox.currentIndex >= listModel.count)
                return;

            var currentData = listModel.get(comboBox.currentIndex);
            if(currentData) {
                var source = listModel.get(comboBox.currentIndex).filePath;

                if(root.currentContentName === comboBox.currentText && null !== loaderLocation.contentItem)
                    return;

                root.currentContentName = comboBox.currentText;

                if(loaderLocation.contentItem) {
                    if(undefined !== loaderLocation.contentItem.setNoSettings)
                        loaderLocation.contentItem.setNoSettings();

                    loaderLocation.contentItem.destroy();
                    loaderLocation.contentItem = null;
                }

                var contentComponent = Qt.createComponent(source);
                if(contentComponent.status === Component.Error) {
                    loaderLocation.errorMessage = contentComponent.errorString();
                    refreshStandbyItem();
                } else {
                    if(0 === root.contentUiId)
                        root.contentUiId = globalUiSettings.generateUiId();

                    var content = contentComponent.createObject(loaderLocation, {"uiId": root.contentUiId, "anchors.fill": loaderLocation});

                    if(undefined !== content.uiId)
                        root.contentUiId = Qt.binding(function() {return content.uiId;});
                    else
                    {
                        globalUiSettings.removeUiId(root.contentUiId);
                        root.contentUiId = 0;
                    }

                    loaderLocation.contentItem = content;
                }
            }
        }

        Image {
            id: toolBarMode
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.bottomMargin: 3
            anchors.leftMargin: 16
            source: displayToolbar ? "qrc:///data/icon/minus.png" : "qrc:///data/icon/plus.png"
            width: 12
            height: width
            z: 2

            property bool displayToolbar: false
            MouseArea {
                anchors.fill: parent
                onClicked: toolBarMode.displayToolbar = !toolBarMode.displayToolbar
            }
        }

        Rectangle {
            id: toolBar
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 22
            color: "lightgrey"
            visible: toolBarMode.displayToolbar
            opacity: 0.75
            z: 1

            MouseArea {
                anchors.fill: parent

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 32

                    ComboBox {
                        id: comboBox
                        Layout.preferredWidth: 150
                        Layout.preferredHeight: 20
                        textRole: "fileBaseName"
                        model: ListModel {
                            id: listModel
                        }

                        onCurrentIndexChanged: {
                            loaderLocation.refresh();
                        }
                    }
                }
            }
        }

        function refreshStandbyItem() {
            if(contentItem) {
                timer.stop();
                standbyItem.visible = false;
            } else {
                timer.start();
            }
        }

        property string errorMessage
        Rectangle {
            id: standbyItem
            anchors.fill: parent
            color: "#555555"
            visible: false

            Label {
                anchors.fill: parent
                color: "red"
                visible: 0 !== loaderLocation.errorMessage.length
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter

                text: "An error occurred, the content could not be loaded ! Reason: " + loaderLocation.errorMessage
                wrapMode: Text.WordWrap
                font.bold: true
            }
        }

        Timer {
            id: timer
            interval: 200
            running: false
            repeat: false
            onTriggered: standbyItem.visible = true
        }
    }
}
