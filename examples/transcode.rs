//! Transcode a video through a pipe-graph pipeline.
//!
//! ```sh
//! cargo run --example transcode -- in.y4m out.y4m
//! cargo run --example transcode -- in.mp4 out.mp4 --clear red --fps 30
//! ```
//!
//! `.y4m` paths use the pure-Rust `y4m_source`/`y4m_sink` nodes; anything else
//! goes through `ffmpeg_source`/`ffmpeg_sink` (which need `ffmpeg` on PATH).
//! `--clear <red|green|blue>` inserts a `clear_channel` stage in between.

use pipe_graph::exec::{Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};

fn spec(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    NodeSpec {
        id: NodeId(id.to_string()),
        kind: kind.to_string(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Params>(),
    }
}

fn port(node: &str, port: &str) -> (NodeId, PortId) {
    (NodeId(node.to_string()), PortId(port.to_string()))
}

fn is_y4m(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".y4m")
}

fn build(input: &str, output: &str, clear: Option<&str>, fps: Option<&str>) -> Graph {
    let mut g = Graph::new();
    let src_kind = if is_y4m(input) {
        "y4m_source"
    } else {
        "ffmpeg_source"
    };
    let sink_kind = if is_y4m(output) {
        "y4m_sink"
    } else {
        "ffmpeg_sink"
    };
    let mut sink_params = vec![("path", output)];
    if let Some(f) = fps {
        sink_params.push(("fps", f));
    }
    // Fresh graph with unique ids: these cannot fail.
    g.add_node(spec("src", src_kind, &[("path", input)]))
        .unwrap();
    g.add_node(spec("sink", sink_kind, &sink_params)).unwrap();
    let into_sink = match clear {
        Some(ch) => {
            g.add_node(spec("clear", "clear_channel", &[("channel", ch)]))
                .unwrap();
            g.connect(port("src", "out"), port("clear", "in")).unwrap();
            port("clear", "out")
        }
        None => port("src", "out"),
    };
    g.connect(into_sink, port("sink", "in")).unwrap();
    g
}

/// The input's frame rate as `num/den`, so the output plays at the same
/// speed: from the y4m header, or via `ffprobe` for anything else.
fn input_fps(input: &str) -> Option<String> {
    if is_y4m(input) {
        let file = std::fs::File::open(input).ok()?;
        let h = pipe_graph::io::Y4mReader::new(file).ok()?.header().clone();
        return Some(format!("{}/{}", h.fps_num, h.fps_den));
    }
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=r_frame_rate", "-of", "csv=p=0"])
        .arg(input)
        .output()
        .ok()?;
    let rate = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !rate.is_empty() && rate != "0/0").then_some(rate)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: transcode <in> <out> [--clear red|green|blue] [--fps N]";
    let (mut positional, mut clear, mut fps) = (Vec::new(), None, None);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--clear" => clear = Some(args.next().ok_or(usage)?),
            "--fps" => fps = Some(args.next().ok_or(usage)?),
            _ => positional.push(a),
        }
    }
    let [input, output] = positional.as_slice() else {
        return Err(usage.into());
    };

    let fps = fps.or_else(|| input_fps(input));
    let g = build(input, output, clear.as_deref(), fps.as_deref());
    let mut rt = Runtime::instantiate(&g, &builtin_registry())?;
    let frames = rt.run_until_eos(u64::MAX)?;
    // Dropping the runtime finalizes the sink (closes ffmpeg and waits for
    // it). A failure there can only be reported on stderr, so double-check
    // that something was actually written before claiming success.
    drop(rt);
    if std::fs::metadata(output).map_or(true, |m| m.len() == 0) {
        return Err(format!("no output was written to {output}").into());
    }
    println!("transcoded {frames} frames: {input} -> {output}");
    Ok(())
}
