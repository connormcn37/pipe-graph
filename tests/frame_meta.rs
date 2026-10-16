//! Stream metadata (A2) and audio stubs (A8).
//!
//! Sources stamp frames with a timestamp, frame number and colour space; the
//! runtime carries that through stages that never mention it; and audio is a
//! payload graphs can carry and validate, though nothing processes it yet.

use std::sync::Arc;
use std::time::Duration;

use pipe_graph::data::{AudioBuffer, ColorSpace, Frame, Meta, Payload, PayloadKind};
use pipe_graph::exec::{
    ExecMode, Inputs, Node, NodeError, Outputs, PortSet, PortSpec, Registry, Runtime,
    ScheduleError, builtin_registry,
};
use pipe_graph::graph::{Graph, NodeId, NodeSpec, Params, PortId};
use pipe_graph::io::{Chroma, Y4mHeader, Y4mWriter};

fn node(id: &str, kind: &str, params: &[(&str, &str)]) -> NodeSpec {
    NodeSpec {
        id: NodeId(id.into()),
        kind: kind.into(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Params>(),
    }
}

fn port(n: &str, p: &str) -> (NodeId, PortId) {
    (NodeId(n.into()), PortId(p.into()))
}

fn id(s: &str) -> NodeId {
    NodeId(s.into())
}

/// A 4x2 grey 25 fps clip of `frames` frames, written to a unique temp file.
fn clip(name: &str, frames: u8) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("pipe-graph-meta-{name}-{}.y4m", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    let mut w = Y4mWriter::new(file, Y4mHeader::new(4, 2, (25, 1), Chroma::C444));
    for i in 0..frames {
        w.write_frame(&Frame::from_rgb8(4, 2, vec![(i * 10, i * 10, i * 10); 8]))
            .unwrap();
    }
    w.flush().unwrap();
    path
}

#[test]
fn source_timing_survives_stages_that_never_mention_it() {
    // y4m -> crop (builds a fresh frame) -> split -> merge (multi-input)
    let path = clip("chain", 3);
    let mut g = Graph::new();
    g.add_node(node(
        "src",
        "y4m_source",
        &[("path", path.to_str().unwrap())],
    ))
    .unwrap();
    g.add_node(node(
        "crop",
        "crop",
        &[("x", "1"), ("y", "0"), ("w", "2"), ("h", "2")],
    ))
    .unwrap();
    g.add_node(node("sp", "split", &[("channels", "3")]))
        .unwrap();
    g.add_node(node("m", "merge", &[("channels", "3")]))
        .unwrap();
    g.connect(port("src", "out"), port("crop", "in")).unwrap();
    g.connect(port("crop", "out"), port("sp", "in")).unwrap();
    for c in 0..3 {
        g.connect(port("sp", &format!("out{c}")), port("m", &format!("in{c}")))
            .unwrap();
    }

    // Fused (in place), unfused, and level-parallel all carry it.
    for (mode, fusion) in [
        (ExecMode::Serial, true),
        (ExecMode::Serial, false),
        (ExecMode::Parallel { threads: 2 }, true),
    ] {
        let mut rt = Runtime::instantiate(&g, &builtin_registry()).unwrap();
        rt.set_exec_mode(mode);
        rt.set_fusion(fusion);
        rt.watch(&id("crop"), "out");
        for frame in 0..3u64 {
            rt.run_once().unwrap();
            for n in ["crop", "m"] {
                let meta = *rt.output(&id(n), "out").unwrap().meta().unwrap();
                let at = format!("{n} {mode:?} fusion={fusion}");
                assert_eq!(meta.frame_index, Some(frame), "{at}");
                assert_eq!(meta.pts, Some(Duration::from_millis(40 * frame)), "{at}");
                assert_eq!(meta.duration, Some(Duration::from_millis(40)), "{at}");
                assert_eq!(meta.color, ColorSpace::Srgb, "{at}");
            }
        }
    }
    std::fs::remove_file(path).ok();
}

/// Emits a 1x1 frame; optionally stamps its own metadata, optionally
/// republishes one cached `Arc`.
struct Emit {
    own: Option<Meta>,
    cached: Option<Arc<Payload>>,
    cache: bool,
}

impl Node for Emit {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let mut frame = Frame::from_rgb8(1, 1, vec![(5, 5, 5)]);
        if let Some(meta) = self.own {
            frame = frame.with_meta(meta);
        }
        if self.cache {
            let shared = self
                .cached
                .get_or_insert_with(|| Arc::new(Payload::Frame(frame)))
                .clone();
            outputs.set_shared("out", shared);
        } else {
            outputs.set("out", Payload::Frame(frame));
        }
        Ok(())
    }
}

fn emit_runtime(own: Option<Meta>, cache: bool) -> Runtime {
    let mut g = Graph::new();
    g.add_node(node("e", "emit", &[])).unwrap();
    let mut reg = Registry::new();
    reg.register("emit", move |_| {
        Ok(Box::new(Emit {
            own,
            cached: None,
            cache,
        }) as Box<dyn Node>)
    });
    let mut rt = Runtime::instantiate(&g, &reg).unwrap();
    let input = Frame::from_rgb8(1, 1, vec![(0, 0, 0)]).with_meta(Meta::for_frame(9, 25, 1));
    rt.set_input(&id("e"), "in", Payload::Frame(input));
    rt
}

#[test]
fn unset_outputs_inherit_and_own_metadata_wins() {
    let mut rt = emit_runtime(None, false);
    rt.run_once().unwrap();
    let meta = *rt.output(&id("e"), "out").unwrap().meta().unwrap();
    assert_eq!(meta.frame_index, Some(9), "inherited from the input");

    let own = Meta {
        frame_index: Some(42),
        color: ColorSpace::Linear,
        ..Meta::default()
    };
    let mut rt = emit_runtime(Some(own), false);
    rt.run_once().unwrap();
    assert_eq!(*rt.output(&id("e"), "out").unwrap().meta().unwrap(), own);
}

#[test]
fn a_cached_shared_output_is_never_stamped() {
    // Stamping would mean mutating (or copying) a value the node holds on to.
    let mut rt = emit_runtime(None, true);
    rt.tick(2).unwrap();
    let out = rt.output_arc(&id("e"), "out").unwrap();
    assert!(out.meta().unwrap().is_unset());
}

/// Produces a stereo buffer on an audio port.
struct Tone;

impl Node for Tone {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Audio)])
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        outputs.set(
            "out",
            Payload::Audio(AudioBuffer::new(48_000, 2, vec![0.25f32; 960])),
        );
        Ok(())
    }
}

/// Forwards whatever arrives on an audio port, untouched.
struct AudioPass;

impl Node for AudioPass {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Audio)],
            vec![PortSpec::new("out", PayloadKind::Audio)],
        )
    }

    fn eval(&mut self, inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let value = inputs
            .get_shared("in")
            .ok_or_else(|| NodeError::MissingInput("in".into()))?;
        outputs.set_shared("out", value);
        Ok(())
    }
}

fn audio_registry() -> Registry {
    let mut reg = builtin_registry();
    reg.register("tone", |_| Ok(Box::new(Tone) as Box<dyn Node>));
    reg.register("audio_pass", |_| Ok(Box::new(AudioPass) as Box<dyn Node>));
    reg
}

#[test]
fn audio_flows_through_a_graph_untouched() {
    let mut g = Graph::new();
    g.add_node(node("tone", "tone", &[])).unwrap();
    g.add_node(node("pass", "audio_pass", &[])).unwrap();
    g.connect(port("tone", "out"), port("pass", "in")).unwrap();

    let mut rt = Runtime::instantiate(&g, &audio_registry()).unwrap();
    rt.watch(&id("tone"), "out");
    rt.run_once().unwrap();
    let audio = rt.output(&id("pass"), "out").unwrap().as_audio().unwrap();
    assert_eq!(
        (audio.sample_rate, audio.channels, audio.frames()),
        (48_000, 2, 480)
    );
    // Forwarded, not copied.
    assert!(Arc::ptr_eq(
        &rt.output_arc(&id("tone"), "out").unwrap(),
        &rt.output_arc(&id("pass"), "out").unwrap()
    ));
}

#[test]
fn audio_cannot_be_wired_into_a_frame_port() {
    let mut g = Graph::new();
    g.add_node(node("tone", "tone", &[])).unwrap();
    g.add_node(node("inv", "invert", &[])).unwrap();
    g.connect(port("tone", "out"), port("inv", "in")).unwrap();

    let err = Runtime::instantiate(&g, &audio_registry())
        .err()
        .expect("audio -> frame must not validate");
    assert!(
        matches!(err, ScheduleError::Validation(_)),
        "unexpected error: {err}"
    );
}
