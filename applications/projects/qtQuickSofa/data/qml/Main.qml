import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import SofaBasics 1.0
import SofaTools 1.0
import SofaWidgets 1.0

Window {
    width: 1280
    height: 720

    Component.onCompleted: {
        visible = true;
    }

    menuBar: DefaultMenuBar {
        id: menuBar
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Component {
            id: dynamicContentComponent

            DynamicContent {
                id: dynamicContent
                defaultContentName: "Viewer"
                sourceDir: "qrc:/SofaWidgets"
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

    statusBar: DefaultStatusBar {
        id: statusBar
    }
}
