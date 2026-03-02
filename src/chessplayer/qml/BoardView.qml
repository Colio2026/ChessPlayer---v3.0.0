import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    // Injected from Python via rootContext().setContextProperty(...)
    // - piecesModel: QAbstractListModel with roles: file, rank, image
    // - bridge: QObject with slots: attemptMove(fromFile,fromRank,toFile,toRank), choosePromotion(promo)

    // Make root always occupy the QQuickWidget
    anchors.fill: parent
    width: parent ? parent.width : 800
    height: parent ? parent.height : 800

    // Square board that never clips; centered in available space
    readonly property real boardSize: Math.min(width, height)

    Rectangle {
        id: board
        width: root.boardSize
        height: root.boardSize
        anchors.centerIn: parent
        radius: 8
        color: "#111"

        readonly property real cell: width / 8.0

        // Squares
        Grid {
            anchors.fill: parent
            rows: 8
            columns: 8

            Repeater {
                model: 64
                Rectangle {
                    readonly property int f: index % 8
                    readonly property int r: Math.floor(index / 8)
                    width: board.cell
                    height: board.cell
                    color: ((f + r) % 2 === 0) ? "#e6e6e6" : "#6a6a6a"
                }
            }
        }

        // Pieces
        Repeater {
            model: piecesModel

            delegate: Item {
                id: pieceItem

                width: board.cell
                height: board.cell

                // Model roles
                property int file: model.file
                property int rank: model.rank
                property string img: model.image

                // Position (rank 0 at bottom)
                x: file * board.cell
                y: (7 - rank) * board.cell

                // Remember original spot for snap-back
                property real homeX: x
                property real homeY: y

                onXChanged: { /* keep homeX/homeY stable only when model changes */ }
                onYChanged: { /* noop */ }

                Image {
                    anchors.fill: parent
                    source: img
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    cache: true
                }

                MouseArea {
                    anchors.fill: parent
                    drag.target: pieceItem
                    drag.axis: Drag.XAndYAxis
                    drag.minimumX: 0
                    drag.minimumY: 0
                    drag.maximumX: board.width - pieceItem.width
                    drag.maximumY: board.height - pieceItem.height

                    onPressed: {
                        pieceItem.z = 10
                        pieceItem.homeX = pieceItem.x
                        pieceItem.homeY = pieceItem.y
                    }

                    onReleased: {
                        pieceItem.z = 0

                        // Nearest square
                        var nx = Math.round(pieceItem.x / board.cell)
                        var ny = Math.round(pieceItem.y / board.cell)
                        nx = Math.max(0, Math.min(7, nx))
                        ny = Math.max(0, Math.min(7, ny))

                        var toFile = nx
                        var toRank = 7 - ny

                        // snap back immediately; Python will refresh the model if legal
                        pieceItem.x = pieceItem.homeX
                        pieceItem.y = pieceItem.homeY

                        if (bridge) {
                            bridge.attemptMove(pieceItem.file, pieceItem.rank, toFile, toRank)
                        }
                    }
                }

                // Smooth when Python refreshes model (piece jumps to new x/y)
                Behavior on x { NumberAnimation { duration: 120 } }
                Behavior on y { NumberAnimation { duration: 120 } }
            }
        }
    }

    // Promotion popup
    Popup {
        id: promoPopup
        modal: true
        focus: true
        closePolicy: Popup.NoAutoClose
        x: (root.width - width) / 2
        y: (root.height - height) / 2
        width: 260
        height: 120

        Rectangle {
            anchors.fill: parent
            radius: 10
            color: "#222"
            border.color: "#444"

            Column {
                anchors.centerIn: parent
                spacing: 10

                Text {
                    text: "Promotion"
                    color: "#eee"
                    font.pixelSize: 16
                    width: parent.width
                    horizontalAlignment: Text.AlignHCenter
                }

                Row {
                    spacing: 10
                    Repeater {
                        model: [
                            { label: "Q", v: "q" },
                            { label: "R", v: "r" },
                            { label: "B", v: "b" },
                            { label: "N", v: "n" }
                        ]
                        delegate: Button {
                            text: modelData.label
                            onClicked: {
                                promoPopup.close()
                                if (bridge) bridge.choosePromotion(modelData.v)
                            }
                        }
                    }
                }
            }
        }
    }

    Connections {
        target: bridge
        function onPromotionRequested(prefix) {
            promoPopup.open()
        }
    }
}