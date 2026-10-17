//! The `pipe-graph` binary's real subcommands: load a graph from the text
//! format (see `graph::Graph::to_text`/`from_text`), then either run it
//! headlessly or just validate it.
//!
//! ```text
//! pipe-graph run <file> [ticks] [--input <node>.<port>=<W>x<H>x<C>[:<value>]]...
//! pipe-graph check <file>
//! ```

use crate::data::{Frame, FrameData, Payload};
use crate::exec::{Runtime, builtin_registry, validate};
use crate::graph::{Graph, NodeId};

const USAGE: &str = "usage:\n  \
    pipe-graph run <file> [ticks] [--input <node>.<port>=<W>x<H>x<C>[:<value>]]...\n  \
    pipe-graph check <file>";

/// Dispatches on `args[0]` (the subcommand); `args` excludes the program name.
pub fn run(args: &[String]) -> Result<(), String> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(USAGE.to_string());
    };
    match cmd.as_str() {
        "run" => cmd_run(rest),
        "check" => cmd_check(rest),
        other => Err(format!("unknown subcommand '{other}'\n{USAGE}")),
    }
}

struct InputArg {
    node: NodeId,
    port: String,
    frame: Frame,
}

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
        Payload::Audio(a) => format!(
            "Audio {} Hz x{} ch, {} frames",
            a.sample_rate,
            a.channels,
            a.frames()
        ),
    }
}

fn load_graph(file: &str) -> Result<Graph, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    Graph::from_text(&text).map_err(|e| format!("{file}: {e}"))
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let run_usage =
        "usage: pipe-graph run <file> [ticks] [--input <node>.<port>=<W>x<H>x<C>[:<value>]]...";
    let mut file = None;
    let mut ticks = None;
    let mut inputs = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--input" {
            let v = it.next().ok_or("--input needs a value")?;
            inputs.push(parse_input(v)?);
        } else if let Some(v) = a.strip_prefix("--input=") {
            inputs.push(parse_input(v)?);
        } else if file.is_none() {
            file = Some(a.clone());
        } else if ticks.is_none() {
            ticks = Some(
                a.parse::<u32>()
                    .map_err(|_| format!("bad tick count '{a}'"))?,
            );
        } else {
            return Err(format!("unexpected argument '{a}'\n{run_usage}"));
        }
    }
    let file = file.ok_or(run_usage)?;
    let ticks = ticks.unwrap_or(1);

    let graph = load_graph(&file)?;
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

fn cmd_check(args: &[String]) -> Result<(), String> {
    let [file] = args else {
        return Err("usage: pipe-graph check <file>".to_string());
    };
    let graph = load_graph(file)?;
    let reg = builtin_registry();
    validate(&graph, &reg).map_err(|e| format!("{file}: {e}"))?;
    println!(
        "ok: {file}: {} nodes, {} edges",
        graph.nodes.len(),
        graph.edges.len()
    );
    Ok(())
}
