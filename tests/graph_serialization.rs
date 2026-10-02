//! Round-trip and error-case tests for the `Graph` text format.

use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, ParseError, ParseErrorKind, PortId};

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
# pipe-graph text format v1
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
        ("= a", 1, ParseErrorKind::Expected("'node' or 'edge'")),
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
