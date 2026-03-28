import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    // Injected from Python via rootContext().setContextProperty(...)
    // - piecesModel      : QAbstractListModel  roles: file, rank, image
    // - bridge           : QObject  slots: attemptMove, choosePromotion
    // - squareIndicators : list of { file, rank, type }
    //     type values: "weak" | "strong" | "tactic" | "king_danger"
    //     Each type maps to a fixed corner and colour (defined below).
    //     Main board always receives [] — dots only appear on the coach board.

    anchors.fill: parent
    width:  parent ? parent.width  : 800
    height: parent ? parent.height : 800

    // Square board centred in available space
    readonly property real boardSize: Math.min(width, height)

    // ── Type → corner + colour table ─────────────────────────────────────────
    // Extend this table when new indicator types are added.
    //   isLeft  true  → top-left  / bottom-left  corner
    //   isLeft  false → top-right / bottom-right corner
    //   isTop   true  → top-left  / top-right    corner
    //   isTop   false → bottom-left / bottom-right corner
    readonly property var indicatorDefs: ({
        "weak":        { isLeft: true,  isTop: true,  colour: "#FF5722" },
        "strong":      { isLeft: false, isTop: true,  colour: "#66BB6A" },
        "tactic":      { isLeft: true,  isTop: false, colour: "#FFD54F" },
        "king_danger": { isLeft: false, isTop: false, colour: "#AB47BC" }
    })

    Rectangle {
        id: board
        width:  root.boardSize
        height: root.boardSize
        anchors.centerIn: parent
        radius: 8
        color: "#111"

        readonly property real cell: width / 8.0

        // ── Squares ───────────────────────────────────────────────────────────
        Grid {
            anchors.fill: parent
            rows: 8
            columns: 8

            Repeater {
                model: 64
                Rectangle {
                    readonly property int f: index % 8
                    readonly property int r: Math.floor(index / 8)
                    width:  board.cell
                    height: board.cell
                    color: ((f + r) % 2 === 0) ? "#e6e6e6" : "#6a6a6a"
                }
            }
        }

        // ── Pieces ────────────────────────────────────────────────────────────
        Repeater {
            model: piecesModel

            delegate: Item {
                id: pieceItem

                width:  board.cell
                height: board.cell

                property int    file: model.file
                property int    rank: model.rank
                property string img:  model.image

                x: file * board.cell
                y: (7 - rank) * board.cell

                property real homeX: x
                property real homeY: y

                onXChanged: {}
                onYChanged: {}

                Image {
                    anchors.fill: parent
                    source:   img
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    cache:  true
                }

                MouseArea {
                    anchors.fill: parent
                    drag.target:   pieceItem
                    drag.axis:     Drag.XAndYAxis
                    drag.minimumX: 0
                    drag.minimumY: 0
                    drag.maximumX: board.width  - pieceItem.width
                    drag.maximumY: board.height - pieceItem.height

                    onPressed: {
                        pieceItem.z     = 10
                        pieceItem.homeX = pieceItem.x
                        pieceItem.homeY = pieceItem.y
                    }

                    onReleased: {
                        pieceItem.z = 0
                        var nx = Math.max(0, Math.min(7, Math.round(pieceItem.x / board.cell)))
                        var ny = Math.max(0, Math.min(7, Math.round(pieceItem.y / board.cell)))
                        var toFile = nx
                        var toRank = 7 - ny
                        pieceItem.x = pieceItem.homeX
                        pieceItem.y = pieceItem.homeY
                        if (bridge) bridge.attemptMove(pieceItem.file, pieceItem.rank, toFile, toRank)
                    }
                }

                Behavior on x { NumberAnimation { duration: 120 } }
                Behavior on y { NumberAnimation { duration: 120 } }
            }
        }

        // ── Square indicators (corner dots) ───────────────────────────────────
        // Reads the 'squareIndicators' context property directly (no root. prefix)
        // so that setContextProperty() updates trigger re-evaluation.
        Repeater {
            model: squareIndicators

            delegate: Item {
                readonly property var   def:     root.indicatorDefs[modelData.type]
                                                 || { isLeft: true, isTop: true, colour: "#888888" }
                readonly property real  dotSize: board.cell * 0.22
                readonly property real  margin:  3
                readonly property real  baseX:   modelData.file * board.cell
                readonly property real  baseY:   (7 - modelData.rank) * board.cell

                x: def.isLeft ? baseX + margin
                              : baseX + board.cell - dotSize - margin
                y: def.isTop  ? baseY + margin
                              : baseY + board.cell - dotSize - margin

                width:  dotSize
                height: dotSize
                z: 8

                Rectangle {
                    anchors.fill: parent
                    radius:       width / 2
                    color:        def.colour
                    opacity:      0.82
                    border.color: Qt.lighter(def.colour, 1.4)
                    border.width: 1
                }
            }
        }
    }

    // ── Promotion popup ───────────────────────────────────────────────────────
    Popup {
        id: promoPopup
        modal: true
        focus: true
        closePolicy: Popup.NoAutoClose
        x: (root.width  - width)  / 2
        y: (root.height - height) / 2
        width:  260
        height: 120

        Rectangle {
            anchors.fill: parent
            radius: 10
            color:  "#222"
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
        function onPromotionRequested(prefix) { promoPopup.open() }
    }
}
