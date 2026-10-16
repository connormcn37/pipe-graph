//! Round-trip and error-case tests for the `Graph` text format.

use std::collections::HashMap;

use pipe_graph::graph::{
    FORMAT_VERSION, Graph, GraphDocument, NodeId, NodeSpec, Params, ParseError, ParseErrorKind,
    PortId,
};

fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    let mut p = Params::new();
    for (k, v) in params {
        p.insert(k.to_string(), v.to_string());
    }
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: p,
    }
}

fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

type Shape = (
    Vec<(String, String, Vec<(String, String)>)>,
    Vec<(String, String, String, String)>,
);

/// Structural view of a graph: nodes sorted by id (params sorted), edges in
/// EdgeId order. Edge ids themselves are excluded — they are not serialized.
fn shape(g: &Graph) -> Shape {
    let mut nodes: Vec<_> = g
        .nodes
        .values()
        .map(|n| {
            let mut params: Vec<_> = n
                .params
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            params.sort();
            (n.id.0.clone(), n.kind.clone(), params)
        })
        .collect();
    nodes.sort();
    let edges = g
        .edges_in_order()
        .into_iter()
        .map(|(_, c)| {
            (
                c.from.0.0.clone(),
                c.from.1.0.clone(),
                c.to.0.0.clone(),
                c.to.1.0.clone(),
            )
        })
        .collect();
    (nodes, edges)
}

fn split_merge() -> Graph {
    let mut g = Graph::new();
    g.add_node(node("src", "clear_channel", &[("channel", "red")]))
        .unwrap();
    g.add_node(node("split", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("merge", "merge", &[("channels", "3")]))
        .unwrap();
    g.connect(port("src", "out"), port("split", "in")).unwrap();
    // Connect in a non-sorted order to check edge order is preserved.
    g.connect(port("split", "out2"), port("merge", "in0"))
        .unwrap();
    g.connect(port("split", "out1"), port("merge", "in1"))
        .unwrap();
    g.connect(port("split", "out0"), port("merge", "in2"))
        .unwrap();
    g
}

#[test]
fn round_trip_preserves_structure() {
    let g = split_merge();
    let text = g.to_text();
    let back = Graph::from_text(&text).unwrap();
    assert_eq!(shape(&back), shape(&g));
    // Deterministic + idempotent.
    assert_eq!(back.to_text(), text);
    assert_eq!(g.to_text(), text);
}

#[test]
fn output_is_sorted_and_readable() {
    let text = split_merge().to_text();
    let expected = "\
pipe-graph 2
node merge merge channels=3
node split split channels=3
node src clear_channel channel=red
edge src.out -> split.in
edge split.out2 -> merge.in0
edge split.out1 -> merge.in1
edge split.out0 -> merge.in2
";
    assert_eq!(text, expected);
}

#[test]
fn round_trip_with_awkward_names() {
    let weird = [
        "a b",
        "x=y",
        "1.5",
        "say \"hi\"",
        "back\\slash",
        "#hash",
        "",
        "multi\nline\ttab\r",
        "->",
        "node",
        "ünïcode",
    ];
    let mut g = Graph::new();
    for (i, w) in weird.iter().enumerate() {
        let params: Vec<(String, String)> = weird
            .iter()
            .enumerate()
            .map(|(j, k)| (format!("{k}{j}"), w.to_string()))
            .collect();
        let params: Vec<(&str, &str)> = params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        g.add_node(node(&format!("{w}{i}"), w, &params)).unwrap();
    }
    for (i, w) in weird.iter().enumerate() {
        let next = (i + 1) % weird.len();
        g.connect(
            port(&format!("{w}{i}"), w),
            port(&format!("{}{next}", weird[next]), "in.0"),
        )
        .unwrap();
    }

    let text = g.to_text();
    // One line per statement despite embedded newlines.
    assert_eq!(text.lines().count(), 1 + 2 * weird.len());
    assert!(text.starts_with("pipe-graph 2\n"));
    let back = Graph::from_text(&text).unwrap();
    assert_eq!(shape(&back), shape(&g));
    assert_eq!(back.to_text(), text);
}

#[test]
fn empty_graph_round_trips() {
    let g = Graph::new();
    let back = Graph::from_text(&g.to_text()).unwrap();
    assert!(back.nodes.is_empty() && back.edges.is_empty());
    assert!(Graph::from_text("").unwrap().nodes.is_empty());
}

#[test]
fn reload_assigns_fresh_sequential_edge_ids() {
    let mut g = split_merge();
    // Punch a hole in the id sequence.
    let first = g.edges_in_order()[0].0.clone();
    g.disconnect(&first);
    let back = Graph::from_text(&g.to_text()).unwrap();
    let ids: Vec<u64> = back.edges_in_order().iter().map(|(id, _)| id.0).collect();
    assert_eq!(ids, vec![0, 1, 2]);
    assert_eq!(shape(&back), shape(&g));
}

#[test]
fn accepts_comments_blank_lines_crlf_and_forward_refs() {
    let text = "\r\n# header\r\n\
                edge a.out -> \"b node\".in   # trailing comment\r\n\
                \t node a clear_channel channel = red\r\n\
                node \"b node\" cast dtype=f32 \"scale\"=\"0.5\"\n";
    let g: Graph = text.parse().unwrap();
    assert_eq!(g.nodes.len(), 2);
    let b = &g.nodes[&NodeId("b node".into())];
    assert_eq!(b.params["scale"], "0.5");
    assert_eq!(g.nodes[&NodeId("a".into())].params["channel"], "red");
    let edges = g.edges_in_order();
    assert_eq!(edges[0].1.to, port("b node", "in"));
}

fn err(text: &str) -> ParseError {
    Graph::from_text(text).unwrap_err()
}

#[test]
fn rejects_duplicate_node_ids() {
    let e = err("node a cast dtype=f32\n\nnode a crop\n");
    assert_eq!(
        e,
        ParseError {
            line: 3,
            kind: ParseErrorKind::DuplicateNodeId("a".into())
        }
    );
    assert_eq!(e.to_string(), "line 3: duplicate node id 'a'");
}

#[test]
fn rejects_edges_to_unknown_nodes() {
    let e = err("node a cast\n# c\nedge a.out -> ghost.in\n");
    assert_eq!(
        e,
        ParseError {
            line: 3,
            kind: ParseErrorKind::UnknownNode("ghost".into())
        }
    );
}

#[test]
fn rejects_malformed_lines_with_line_numbers() {
    let cases: &[(&str, usize, ParseErrorKind)] = &[
        (
            "bogus a b",
            1,
            ParseErrorKind::UnknownStatement("bogus".into()),
        ),
        (
            "\"node\" a b",
            1,
            ParseErrorKind::UnknownStatement("node".into()),
        ),
        (
            "= a",
            1,
            ParseErrorKind::Expected("'node', 'edge' or 'pos'"),
        ),
        ("\n\nnode", 3, ParseErrorKind::Expected("node id")),
        ("node a", 1, ParseErrorKind::Expected("node kind")),
        (
            "node a k x",
            1,
            ParseErrorKind::Expected("'=' after parameter key"),
        ),
        (
            "node a k x=",
            1,
            ParseErrorKind::Expected("parameter value"),
        ),
        (
            "node a k x=1 x=2",
            1,
            ParseErrorKind::DuplicateParam("x".into()),
        ),
        (
            "node a k\nedge a.out b.in",
            2,
            ParseErrorKind::Expected("'->' between endpoints"),
        ),
        (
            "node a k\nedge a.out \"->\" a.in",
            2,
            ParseErrorKind::Expected("'->' between endpoints"),
        ),
        (
            "node a k\nedge a -> a.in",
            2,
            ParseErrorKind::Expected("'.' between node and port"),
        ),
        (
            "node a k\nedge a.out -> a.in extra",
            2,
            ParseErrorKind::Expected("end of line after edge"),
        ),
        (
            "node a k\nedge a.out->a.in",
            2,
            ParseErrorKind::Expected("'->' between endpoints"),
        ),
        ("node \"a k", 1, ParseErrorKind::UnterminatedString),
        ("node \"a\\q\" k", 1, ParseErrorKind::BadEscape(Some('q'))),
        ("node \"a\"b k", 1, ParseErrorKind::UnexpectedChar('b')),
    ];
    for (text, line, kind) in cases {
        assert_eq!(
            err(text),
            ParseError {
                line: *line,
                kind: kind.clone()
            },
            "input: {text:?}"
        );
    }
}

#[test]
fn sample_graph_file_loads_and_runs() {
    use pipe_graph::data::{Frame, FrameData, Payload};
    use pipe_graph::exec::{Runtime, builtin_registry};

    let g = Graph::from_text(include_str!("../examples/graphs/split_merge.graph")).unwrap();
    assert_eq!(shape(&Graph::from_text(&g.to_text()).unwrap()), shape(&g));

    let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
    rt.set_input(
        &NodeId("src".into()),
        "in",
        Payload::Frame(Frame::from_data(4, 4, 3, FrameData::U8(vec![200; 48]))),
    );
    rt.run_once().unwrap();
    let out = rt.output(&NodeId("crop".into()), "out").unwrap();
    let f = out.as_frame().unwrap();
    assert_eq!((f.width, f.height, f.channels), (2, 2, 3));
    // Green cleared, then R/B swapped (both 200): every pixel is (200, 0, 200).
    assert_eq!(f.to_rgb8(), vec![(200, 0, 200); 4]);
}

// ---- v2: version header and layout ----------------------------------------

fn doc_err(text: &str) -> ParseError {
    GraphDocument::from_text(text).unwrap_err()
}

fn id(s: &str) -> NodeId {
    NodeId(s.to_string())
}

#[test]
fn writer_emits_current_version_header() {
    assert_eq!(FORMAT_VERSION, 2);
    assert_eq!(Graph::new().to_text(), "pipe-graph 2\n");
    assert_eq!(GraphDocument::new(Graph::new()).to_text(), "pipe-graph 2\n");
}

#[test]
fn v1_files_still_load() {
    // The committed v1 sample has no header (only a comment) and must load
    // unchanged, with no layout.
    let v1 = include_str!("../examples/graphs/split_merge.graph");
    assert!(!v1.lines().any(|l| l.starts_with("pipe-graph")));
    let doc = GraphDocument::from_text(v1).unwrap();
    assert_eq!(doc.graph.nodes.len(), 4);
    assert_eq!(doc.graph.edges.len(), 5);
    assert!(doc.positions.is_empty());
    // An explicit v1 header is accepted too, after comments and blank lines.
    let g = Graph::from_text("# c\n\npipe-graph 1\nnode a cast\n").unwrap();
    assert_eq!(g.nodes.len(), 1);
}

#[test]
fn v1_and_v2_samples_describe_the_same_graph() {
    let v1 = Graph::from_text(include_str!("../examples/graphs/split_merge.graph")).unwrap();
    let v2 =
        GraphDocument::from_text(include_str!("../examples/graphs/split_merge_v2.graph")).unwrap();
    assert_eq!(shape(&v2.graph), shape(&v1));
    assert_eq!(v2.positions.len(), 4);
    assert_eq!(v2.positions[&id("split")], (230.0, -12.5));
}

#[test]
fn v2_sample_round_trips_byte_for_byte() {
    let text = include_str!("../examples/graphs/split_merge_v2.graph");
    let doc = GraphDocument::from_text(text).unwrap();
    assert_eq!(doc.to_text(), text);
    // `Graph::from_text` accepts it too, dropping the layout.
    let g = Graph::from_text(text).unwrap();
    assert!(!g.to_text().contains("\npos "));
}

#[test]
fn positions_round_trip_exactly_and_sorted() {
    let mut doc = GraphDocument::new(split_merge());
    let awkward = [
        ("src", (0.1_f32, -0.0_f32)),
        ("split", (f32::MAX, f32::MIN_POSITIVE)),
        ("merge", (1.0 / 3.0, -1e-30)),
    ];
    for (n, p) in awkward {
        doc.positions.insert(id(n), p);
    }
    let text = doc.to_text();
    let pos_lines: Vec<&str> = text.lines().filter(|l| l.starts_with("pos ")).collect();
    assert_eq!(pos_lines.len(), 3);
    assert!(pos_lines[0].starts_with("pos merge "));
    assert!(pos_lines[1].starts_with("pos split "));
    assert_eq!(pos_lines[2], "pos src 0.1 -0");
    // pos lines come after every node and edge line.
    let first_pos = text.find("\npos ").unwrap();
    assert!(!text[first_pos..].contains("\nnode ") && !text[first_pos..].contains("\nedge "));

    let back = GraphDocument::from_text(&text).unwrap();
    for (n, (x, y)) in awkward {
        let (bx, by) = back.positions[&id(n)];
        assert_eq!(
            (bx.to_bits(), by.to_bits()),
            (x.to_bits(), y.to_bits()),
            "{n}"
        );
    }
    assert_eq!(back.to_text(), text);
}

#[test]
fn writer_skips_unwritable_positions() {
    let mut doc = GraphDocument::new(split_merge());
    doc.positions.insert(id("ghost"), (1.0, 2.0));
    doc.positions.insert(id("src"), (f32::NAN, 0.0));
    doc.positions.insert(id("split"), (3.0, 4.0));
    let text = doc.to_text();
    assert!(text.ends_with("pos split 3 4\n"));
    assert_eq!(text.matches("pos ").count(), 1);
    GraphDocument::from_text(&text).unwrap();
}

#[test]
fn positions_with_quoted_ids_and_forward_refs() {
    let text = "pipe-graph 2\npos \"my node\" 1.5 -2 # c\nnode \"my node\" cast\n";
    let doc: GraphDocument = text.parse().unwrap();
    assert_eq!(doc.positions[&id("my node")], (1.5, -2.0));
    assert_eq!(
        doc.to_text(),
        "pipe-graph 2\nnode \"my node\" cast\npos \"my node\" 1.5 -2\n"
    );
}

#[test]
fn rejects_future_versions() {
    let e = doc_err("# written by a newer build\npipe-graph 3\nnode a cast\n");
    assert_eq!(
        e,
        ParseError {
            line: 2,
            kind: ParseErrorKind::UnsupportedVersion(3)
        }
    );
    assert!(e.to_string().contains("version 3 is newer"), "{e}");
    // Graph::from_text rejects it the same way.
    assert_eq!(
        Graph::from_text("pipe-graph 99").unwrap_err().kind,
        ParseErrorKind::UnsupportedVersion(99)
    );
}

#[test]
fn rejects_bad_v2_input_with_line_numbers() {
    let requires_v2 = ParseErrorKind::RequiresVersion {
        statement: "pos",
        version: 2,
    };
    let cases: &[(&str, usize, ParseErrorKind)] = &[
        ("node a k\npipe-graph 2", 2, ParseErrorKind::MisplacedHeader),
        (
            "pipe-graph 2\npipe-graph 2",
            2,
            ParseErrorKind::MisplacedHeader,
        ),
        ("pipe-graph", 1, ParseErrorKind::Expected("format version")),
        (
            "pipe-graph two",
            1,
            ParseErrorKind::BadVersion("two".into()),
        ),
        ("pipe-graph 0", 1, ParseErrorKind::BadVersion("0".into())),
        ("pipe-graph +2", 1, ParseErrorKind::BadVersion("+2".into())),
        (
            "pipe-graph 2.0",
            1,
            ParseErrorKind::BadVersion("2.0".into()),
        ),
        (
            "pipe-graph \"2\"",
            1,
            ParseErrorKind::Expected("format version"),
        ),
        (
            "pipe-graph 2\nnode a k\npos a.5 3",
            3,
            ParseErrorKind::Expected("x coordinate"),
        ),
        // Dangling references are reported in file order.
        (
            "pipe-graph 2\nnode a k\npos ghost 1 2\nedge nope.out -> a.in",
            3,
            ParseErrorKind::UnknownPositionNode("ghost".into()),
        ),
        (
            "pipe-graph 2\nnode a k\nedge nope.out -> a.in\npos ghost 1 2",
            3,
            ParseErrorKind::UnknownNode("nope".into()),
        ),
        (
            "pipe-graph 2 extra",
            1,
            ParseErrorKind::Expected("end of line after version header"),
        ),
        ("node a k\npos a 1 2", 2, requires_v2.clone()),
        ("pipe-graph 1\nnode a k\npos a 1 2", 3, requires_v2),
        (
            "pipe-graph 2\nnode a k\n\npos ghost 1 2",
            4,
            ParseErrorKind::UnknownPositionNode("ghost".into()),
        ),
        (
            "pipe-graph 2\nnode a k\npos a 1 2\npos a 3 4",
            4,
            ParseErrorKind::DuplicatePosition("a".into()),
        ),
        (
            "pipe-graph 2\nnode a k\npos a x 2",
            3,
            ParseErrorKind::BadNumber("x".into()),
        ),
        (
            "pipe-graph 2\nnode a k\npos a 1.2.3 2",
            3,
            ParseErrorKind::BadNumber("1.2.3".into()),
        ),
        (
            "pipe-graph 2\nnode a k\npos a inf 2",
            3,
            ParseErrorKind::BadNumber("inf".into()),
        ),
        (
            "pipe-graph 2\nnode a k\npos a 1",
            3,
            ParseErrorKind::Expected("y coordinate"),
        ),
        (
            "pipe-graph 2\nnode a k\npos a \"1\" 2",
            3,
            ParseErrorKind::Expected("x coordinate"),
        ),
        (
            "pipe-graph 2\nnode a k\npos a 1 2 3",
            3,
            ParseErrorKind::Expected("end of line after pos"),
        ),
        (
            "pipe-graph 2\nnode a k\npos a 1 . 5",
            3,
            ParseErrorKind::BadNumber(".".into()),
        ),
        (
            "pipe-graph 2\npos",
            2,
            ParseErrorKind::Expected("node id after 'pos'"),
        ),
    ];
    for (text, line, kind) in cases {
        assert_eq!(
            doc_err(text),
            ParseError {
                line: *line,
                kind: kind.clone()
            },
            "input: {text:?}"
        );
    }
    assert_eq!(
        doc_err("pipe-graph 2\nnode a k\npos a 1 2\npos a 3 4").to_string(),
        "line 4: duplicate position for node 'a'"
    );
}

#[test]
fn live_session_saves_and_restores_layout() {
    use pipe_graph::editor::{EditorCommand, LiveSession};
    use pipe_graph::exec::builtin_registry;

    let g = Graph::from_text(include_str!("../examples/graphs/split_merge.graph")).unwrap();
    let mut session = LiveSession::with_graph(g, builtin_registry());
    session.apply(EditorCommand::MoveNode {
        node: id("merge"),
        to: (123.25, -45.5),
    });
    let saved = session.to_document().to_text();
    assert!(saved.contains("pos merge 123.25 -45.5\n"), "{saved}");

    let restored = LiveSession::from_document(
        GraphDocument::from_text(&saved).unwrap(),
        builtin_registry(),
    );
    assert!(restored.last_error().is_none());
    assert_eq!(shape(restored.graph()), shape(session.graph()));
    let layout_of = |s: &LiveSession| -> HashMap<NodeId, (f32, f32)> {
        s.layout().iter().map(|(k, v)| (k.clone(), v)).collect()
    };
    assert_eq!(layout_of(&restored), layout_of(&session));
    // Saving the restored session reproduces the file.
    assert_eq!(restored.to_document().to_text(), saved);
}

#[test]
fn live_session_auto_places_nodes_missing_from_the_file() {
    use pipe_graph::editor::LiveSession;
    use pipe_graph::exec::builtin_registry;

    // Only `src` has a saved position; it sits where auto-layout would put
    // `split`, so the auto-placed nodes must avoid it.
    let text = include_str!("../examples/graphs/split_merge.graph")
        .replace("node src", "pipe-graph 2\nnode src")
        + "pos src 220 0\n";
    let doc = GraphDocument::from_text(&text).unwrap();
    let session = LiveSession::from_document(doc, builtin_registry());
    let layout = session.layout();
    assert_eq!(layout.len(), 4);
    assert_eq!(layout.position(&id("src")), Some((220.0, 0.0)));
    let split = layout.position(&id("split")).unwrap();
    assert_ne!(split, (220.0, 0.0));

    // A v1 file restores with a full auto-layout.
    let v1 =
        GraphDocument::from_text(include_str!("../examples/graphs/split_merge.graph")).unwrap();
    let s = LiveSession::from_document(v1, builtin_registry());
    assert_eq!(s.layout().len(), 4);
}
