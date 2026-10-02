//! Load a graph from the text format, run it, and print what it produced.
//!
//! ```text
//! cargo run --example run_graph -- <file> [ticks] [--input <node>.<port>=<W>x<H>x<C>[:<value>]]...
//! ```
//!
//! * `<file>` — a graph in the `Graph::to_text` format
//!   (e.g. `examples/graphs/split_merge.graph`).
//! * `[ticks]` — how many times to run the graph (default 1).
//! * `--input node.port=WxHxC:V` — feed a synthetic `u8` frame of the given
//!   shape, every sample set to `V`, into a source node's input port. Repeat
//!   for several sources. `:V` may be omitted (defaults to 0). The node id is
//!   everything before the *last* `.`, so ids containing dots still work.
//!
//! Every output port the runtime captured (a pipeline's results) is printed
//! with its shape, in node-id order.

use std::process::ExitCode;

use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId};

struct InputArg {
    node: NodeId,
    port: String,
    frame: Frame,
}

const USAGE: &str =
    "usage: run_graph <file> [ticks] [--input <node>.<port>=<W>x<H>x<C>[:<value>]]...";

fn parse_input(arg: &str) -> Result<InputArg, String> {
    let usage = || format!("bad --input '{arg}': expected <node>.<port>=<W>x<H>x<C>[:<value>]");
    let (target, spec) = arg.rsplit_once('=').ok_or_else(usage)?;
    let (node, port) = target.rsplit_once('.').ok_or_else(usage)?;
    let (dims, value) = match spec.split_once(':') {
        Some((d, v)) => (d, v.parse::<u8>().map_err(|_| usage())?),
        None => (spec, 0),
    };
    let dims: Vec<u32> = dims
        .split('x')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| usage())?;
    let [w, h, c] = dims[..] else {
        return Err(usage());
    };
    let len = w as usize * h as usize * c as usize;
    Ok(InputArg {
        node: NodeId(node.to_string()),
        port: port.to_string(),
        frame: Frame::from_data(w, h, c, FrameData::U8(vec![value; len])),
    })
}

fn describe(p: &Payload) -> String {
    match p {
        Payload::Frame(f) => format!(
            "Frame {}x{}x{} {:?}",
            f.width,
            f.height,
            f.channels,
            f.dtype()
        ),
        Payload::Scalar(v) => format!("Scalar {v}"),
        Payload::Bytes(b) => format!("Bytes len={}", b.len()),
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut ticks = None;
    let mut inputs = Vec::new();
    while let Some(a) = args.next() {
        if a == "--input" {
            let v = args.next().ok_or("--input needs a value")?;
            inputs.push(parse_input(&v)?);
        } else if let Some(v) = a.strip_prefix("--input=") {
            inputs.push(parse_input(v)?);
        } else if file.is_none() {
            file = Some(a);
        } else if ticks.is_none() {
            ticks = Some(
                a.parse::<u32>()
                    .map_err(|_| format!("bad tick count '{a}'"))?,
            );
        } else {
            return Err(format!("unexpected argument '{a}'\n{USAGE}"));
        }
    }
    let file = file.ok_or(USAGE)?;
    let ticks = ticks.unwrap_or(1);

    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let graph = Graph::from_text(&text).map_err(|e| format!("{file}: {e}"))?;
    let reg = builtin_registry();
    let mut rt = Runtime::instantiate(&graph, &reg).map_err(|e| format!("{e:?}"))?;
    for i in inputs {
        if !graph.nodes.contains_key(&i.node) {
            return Err(format!("--input names unknown node '{}'", i.node.0));
        }
        rt.set_input(&i.node, &i.port, Payload::Frame(i.frame));
    }
    rt.tick(ticks).map_err(|e| format!("{e:?}"))?;

    println!(
        "ran {file}: {} nodes, {} edges, {ticks} tick(s)",
        graph.nodes.len(),
        graph.edges.len()
    );
    let mut ids: Vec<&NodeId> = graph.nodes.keys().collect();
    ids.sort_by(|a, b| a.0.cmp(&b.0));
    for id in ids {
        let ports = reg
            .ports_of(&graph.nodes[id])
            .map_err(|e| format!("{e:?}"))?;
        for out in &ports.outputs {
            if let Some(p) = rt.output(id, &out.id.0) {
                println!("{}.{}: {}", id.0, out.id.0, describe(p));
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
