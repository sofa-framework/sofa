import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import SofaBasics 1.0
import SofaTools 1.0
import SofaWidgets 1.0

Window {
    id: root
    width: 1280
    height: 720

    Component.onCompleted: {
        visible = true;
    }

    menuBar: DefaultMenuBar {
        id: menuBar
    }

    DynamicSplitView {
        id: dynamicSplitView
        anchors.fill: parent
        uiId: 1
        sourceComponent: Component {
            DynamicContent {
                defaultContentName: "Viewer"
                sourceDir: "qrc:/SofaWidgets"
            }
        }
    }

    statusBar: DefaultStatusBar {
        id: statusBar
        statusMessage: root.statusMessage
    }
}
