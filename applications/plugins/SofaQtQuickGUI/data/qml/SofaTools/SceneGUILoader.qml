import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.0
import SofaBasics 1.0
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

ContentItem {
    id: root

    property string title: "Scene GUI"

    property int priority: 100
    property var scene
    property url source: scene ? scene.sourceQML : ""
    readonly property alias status: d.status
    readonly property alias item: d.item

    QtObject {
        id: d

        property Item item
        property Component componentFactory
        property int status: Loader.Null

        property Timer timer: Timer {
            running: false
            repeat: false
            interval: 1

            onTriggered: {
                errorLabel.text = "";

                // use a fresh version of the gui if it's a reload by removing the old version of the cache
                SofaToolsScript.Tools.trimCache();

                if(0 !== root.source.toString().length) {
                    d.componentFactory = Qt.createComponent(root.source);
                    if(Component.Ready === d.componentFactory.status)
                        d.item = d.componentFactory.createObject(layout, {"Layout.fillWidth": true});

                    if(!d.item) {
                        errorLabel.text = "Cannot create Component from:" + root.source + "\n\n";
                        errorLabel.text += d.componentFactory.errorString().replace("\n", "\n\n");
                        d.status = Loader.Error;
                        return;
                    }

                    d.status = Loader.Ready;
                }
            }
        }
    }

    onSourceChanged: {
        if(d.item)
            d.item.destroy();

        // we have to do this to be able to trim the item from cache
        if(d.componentFactory)
            d.componentFactory.destroy();

        if(0 !== root.source.toString().length)
            d.status = Loader.Loading;
        else
            d.status = Loader.Null;

        // delay loading of the component to the next frame to let qml completely destroy the previous one allowing us to trim it from cache
        d.timer.start();
    }

    ColumnLayout {
        id: layout
        anchors.fill: parent
        spacing: 0

//        Item {
//            Layout.fillWidth: true
//            implicitHeight: busyIndicator.implicitHeight
//            visible: busyIndicator.running

//            BusyIndicator {
//                id: busyIndicator
//                anchors.horizontalCenter: parent.horizontalCenter
//                anchors.top: parent.top
//                implicitWidth: 100
//                implicitHeight: implicitWidth
//                running: root.item ? false : Loader.Loading === root.status
//            }
//        }

        GroupBox {
            Layout.fillWidth: true
            implicitWidth: 0
            visible: Loader.Error === status
            title: "Error Loading Scene GUI"

            ColumnLayout {
                anchors.fill: parent

                TextArea {
                    id: errorLabel
                    Layout.fillWidth: true
                    Layout.preferredHeight: 250
                    wrapMode: Text.WordWrap
                    readOnly: true
                }
            }
        }
    }
}
