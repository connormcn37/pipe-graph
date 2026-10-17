//! CPU/GPU conformance (H3): every stage with a GPU implementation must give
//! the same results as its CPU implementation — bit for bit unless its case
//! names a tolerance — and the same errors, whether its inputs start on the
//! CPU or already on the GPU.
//!
//! Adding a GPU path to a stage means adding cases here:
//! `every_gpu_capable_kind_has_cases` fails until there are some.
#![cfg(feature = "gpu")]

use std::collections::{BTreeSet, HashMap};

use pipe_graph::data::{DType, Frame, FrameData, Location, Payload};
use pipe_graph::exec::{Inputs, Node, Outputs, builtin_registry};
use pipe_graph::gpu::GpuContext;
use pipe_graph::graph::{NodeId, NodeSpec, Params, PortId};

/// A context, or `None` (test skipped) without an adapter — unless
/// `PIPE_GRAPH_REQUIRE_GPU` is set, as in CI.
fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) if std::env::var_os("PIPE_GRAPH_REQUIRE_GPU").is_some() => {
            panic!("PIPE_GRAPH_REQUIRE_GPU is set but {e}")
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// One stage evaluation to run on both paths.
struct Case {
    kind: &'static str,
    params: Vec<(&'static str, String)>,
    inputs: Vec<(&'static str, Frame)>,
    /// Largest allowed `|cpu - gpu|` for `f32` samples; `0.0` means
    /// bit-identical (NaN matching NaN). `u8` is always exact.
    tolerance: f32,
}

fn case(
    kind: &'static str,
    params: &[(&'static str, &str)],
    inputs: Vec<(&'static str, Frame)>,
) -> Case {
    Case {
        kind,
        params: params.iter().map(|(k, v)| (*k, v.to_string())).collect(),
        inputs,
        tolerance: 0.0,
    }
}

impl Case {
    fn within(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    fn label(&self) -> String {
        let shapes: Vec<String> = self
            .inputs
            .iter()
            .map(|(p, f)| {
                format!(
                    "{p}={}x{}x{} {:?}",
                    f.width,
                    f.height,
                    f.channels,
                    f.dtype()
                )
            })
            .collect();
        format!("{} {:?} [{}]", self.kind, self.params, shapes.join(", "))
    }

    fn build(&self) -> Box<dyn Node> {
        let spec = NodeSpec {
            id: NodeId("n".into()),
            kind: self.kind.into(),
            params: self
                .params
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<Params>(),
        };
        builtin_registry().build(&spec).unwrap()
    }
}

/// Deterministic pseudo-random samples (xorshift), so failures reproduce.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn u8_frame(&mut self, w: u32, h: u32, ch: u32) -> Frame {
        let n = (w * h * ch) as usize;
        Frame::from_data(
            w,
            h,
            ch,
            FrameData::U8((0..n).map(|_| self.next() as u8).collect()),
        )
    }

    /// Samples in `[0, 1]`, the range the `f32` pipeline works in.
    fn f32_frame(&mut self, w: u32, h: u32, ch: u32) -> Frame {
        let n = (w * h * ch) as usize;
        let data = (0..n)
            .map(|_| (self.next() % 10_001) as f32 / 10_000.0)
            .collect();
        Frame::from_data(w, h, ch, FrameData::F32(data))
    }
}

/// Shapes covering a single pixel, sizes that are not a whole number of
/// words or workgroups, and several workgroups.
const SHAPES: [(u32, u32); 3] = [(1, 1), (7, 5), (61, 37)];

fn cases() -> Vec<Case> {
    let mut rng = Rng(0x9e37_79b9);
    let mut cases = Vec::new();

    for (w, h) in SHAPES {
        for ch in [1, 3, 4] {
            let (a, b) = (rng.u8_frame(w, h, ch), rng.f32_frame(w, h, ch));
            for f in [&a, &b] {
                cases.push(case("invert", &[], vec![("in", f.clone())]));
                for factor in ["0.5", "1.7", "-1"] {
                    cases.push(case("gain", &[("factor", factor)], vec![("in", f.clone())]));
                }
                for level in ["0", "0.5", "1.0001"] {
                    cases.push(case(
                        "threshold",
                        &[("level", level)],
                        vec![("in", f.clone())],
                    ));
                }
                for r in ["0", "1", "3", "40"] {
                    let c = case("box_blur", &[("radius", r)], vec![("in", f.clone())]);
                    // f32 sums in f32 on the GPU, f64 on the CPU.
                    cases.push(if f.dtype() == DType::F32 {
                        c.within(1e-5)
                    } else {
                        c
                    });
                }
                if ch >= 3 {
                    let c = case("grayscale", &[], vec![("in", f.clone())]);
                    // A fused multiply-add may differ in the last bit.
                    cases.push(if f.dtype() == DType::F32 {
                        c.within(1e-6)
                    } else {
                        c
                    });
                }
                let ch_s = ch.to_string();
                cases.push(case(
                    "split",
                    &[("channels", &ch_s)],
                    vec![("in", f.clone())],
                ));
                cases.push(case(
                    "crop",
                    &[
                        ("x", "0"),
                        ("y", "0"),
                        ("w", &w.to_string()),
                        ("h", &h.to_string()),
                    ],
                    vec![("in", f.clone())],
                ));
                if w > 2 && h > 2 {
                    cases.push(case(
                        "crop",
                        &[
                            ("x", "1"),
                            ("y", "2"),
                            ("w", &(w - 2).to_string()),
                            ("h", &(h - 2).to_string()),
                        ],
                        vec![("in", f.clone())],
                    ));
                }
            }
            cases.push(case("cast", &[("dtype", "f32")], vec![("in", a.clone())]));
            cases.push(case(
                "cast",
                &[("dtype", "f32"), ("scale", "2")],
                vec![("in", a.clone())],
            ));
            cases.push(case("cast", &[("dtype", "u8")], vec![("in", b.clone())]));
            cases.push(case("cast", &[("dtype", "u8")], vec![("in", a.clone())]));
            cases.push(case("cast", &[("dtype", "f32")], vec![("in", b.clone())]));
        }
        // Merge k single-channel frames, both dtypes.
        for k in [1u32, 3, 4] {
            let u: Vec<_> = (0..k).map(|_| rng.u8_frame(w, h, 1)).collect();
            let f: Vec<_> = (0..k).map(|_| rng.f32_frame(w, h, 1)).collect();
            for frames in [u, f] {
                let ports = ["in0", "in1", "in2", "in3"];
                let inputs = ports.iter().copied().zip(frames).collect();
                cases.push(case("merge", &[("channels", &k.to_string())], inputs));
            }
        }
    }

    // Rounding and special values for f32 -> u8: halves round away from zero
    // (not to even), out-of-range clamps, NaN is 0.
    let special = vec![
        0.5,
        1.5,
        2.5,
        254.5,
        255.5,
        -0.5,
        -1.5,
        0.499_999_97,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        1e30,
    ];
    let n = special.len() as u32;
    let special = Frame::from_data(n, 1, 1, FrameData::F32(special));
    cases.push(case(
        "cast",
        &[("dtype", "u8"), ("scale", "1")],
        vec![("in", special.clone())],
    ));
    for kind in ["invert", "split", "crop"] {
        let params: &[(&str, &str)] = match kind {
            "split" => &[("channels", "1")],
            "crop" => &[("x", "1"), ("y", "0"), ("w", "5"), ("h", "1")],
            _ => &[],
        };
        cases.push(case(kind, params, vec![("in", special.clone())]));
    }
    cases.push(case(
        "gain",
        &[("factor", "2")],
        vec![("in", special.clone())],
    ));
    cases.push(case(
        "threshold",
        &[("level", "0.5")],
        vec![("in", special)],
    ));

    // Large enough that one dispatch dimension cannot hold it
    // (more than 65535 * 64 invocations): exercises the 2-D split.
    let big = rng.u8_frame(2400, 1800, 4);
    cases.push(case("invert", &[], vec![("in", big.clone())]));
    cases.push(case("gain", &[("factor", "0.75")], vec![("in", big)]));

    cases
}

/// Same samples, bit for bit (NaN matches NaN), or within `tol` for `f32`.
fn assert_same(label: &str, port: &PortId, cpu: &Frame, gpu: &Frame, tol: f32) {
    let ctx = format!("{label} port {}", port.0);
    assert_eq!(
        (gpu.width, gpu.height, gpu.channels, gpu.dtype()),
        (cpu.width, cpu.height, cpu.channels, cpu.dtype()),
        "{ctx}: shape"
    );
    let gpu = gpu.to_cpu().unwrap();
    match (cpu.data(), &*gpu) {
        (FrameData::U8(c), FrameData::U8(g)) => {
            if let Some(i) = (0..c.len()).find(|&i| c[i] != g[i]) {
                panic!("{ctx}: sample {i}: cpu {} gpu {}", c[i], g[i]);
            }
        }
        (FrameData::F32(c), FrameData::F32(g)) => {
            for (i, (&x, &y)) in c.iter().zip(g).enumerate() {
                let same = x.to_bits() == y.to_bits()
                    || (x.is_nan() && y.is_nan())
                    || (tol > 0.0 && (x - y).abs() <= tol);
                assert!(
                    same,
                    "{ctx}: sample {i}: cpu {x:?} gpu {y:?} (tolerance {tol})"
                );
            }
        }
        _ => unreachable!("dtypes compared above"),
    }
}

fn inputs_of(frames: &[(&'static str, Frame)]) -> Inputs {
    Inputs::new(
        frames
            .iter()
            .map(|(p, f)| (PortId(p.to_string()), Payload::Frame(f.clone())))
            .collect::<HashMap<_, _>>(),
    )
}

#[test]
fn gpu_paths_match_the_cpu() {
    let Some(ctx) = context() else { return };
    let cases = cases();
    for case in &cases {
        let label = case.label();
        let mut cpu_out = Outputs::new();
        case.build()
            .eval(&inputs_of(&case.inputs), &mut cpu_out)
            .unwrap_or_else(|e| panic!("{label}: cpu failed: {e}"));
        let cpu_map = cpu_out.into_map();

        // Inputs starting on the CPU (the stage uploads them) and already on
        // the GPU (it reads them in place).
        let on_gpu: Vec<_> = case
            .inputs
            .iter()
            .map(|(p, f)| (*p, f.clone().into_gpu(&ctx).unwrap()))
            .collect();
        for inputs in [inputs_of(&case.inputs), inputs_of(&on_gpu)] {
            let mut gpu_out = Outputs::new();
            case.build()
                .eval_gpu(&ctx, &inputs, &mut gpu_out)
                .unwrap_or_else(|| panic!("{label}: no GPU path"))
                .unwrap_or_else(|e| panic!("{label}: gpu failed: {e}"));
            let gpu_map = gpu_out.into_map();
            assert_eq!(
                gpu_map.keys().map(|p| &p.0).collect::<BTreeSet<_>>(),
                cpu_map.keys().map(|p| &p.0).collect::<BTreeSet<_>>(),
                "{label}: output ports"
            );
            for (port, cpu) in &cpu_map {
                let gpu = gpu_map[port].as_frame().unwrap();
                if case.kind != "cast"
                    || cpu.as_frame().unwrap().dtype() != case.inputs[0].1.dtype()
                {
                    assert_eq!(
                        gpu.location(),
                        Location::Gpu,
                        "{label}: output stays on the GPU"
                    );
                }
                assert_same(&label, port, cpu.as_frame().unwrap(), gpu, case.tolerance);
            }
        }
    }
    eprintln!("{} conformance cases", cases.len());
}

#[test]
fn gpu_paths_fail_like_the_cpu() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(7);
    let rgb = rng.u8_frame(4, 3, 3);
    let failing = [
        case(
            "crop",
            &[("x", "2"), ("y", "0"), ("w", "3"), ("h", "1")],
            vec![("in", rgb.clone())],
        ),
        case("split", &[("channels", "4")], vec![("in", rgb.clone())]),
        case("grayscale", &[], vec![("in", rng.u8_frame(4, 3, 1))]),
        case(
            "merge",
            &[("channels", "2")],
            vec![
                ("in0", rng.u8_frame(4, 3, 1)),
                ("in1", rng.u8_frame(3, 3, 1)),
            ],
        ),
        case(
            "merge",
            &[("channels", "2")],
            vec![
                ("in0", rng.u8_frame(4, 3, 1)),
                ("in1", rng.f32_frame(4, 3, 1)),
            ],
        ),
        case(
            "merge",
            &[("channels", "2")],
            vec![("in0", rng.u8_frame(4, 3, 1)), ("in1", rgb.clone())],
        ),
        case("invert", &[], vec![]),
    ];
    for case in &failing {
        let label = case.label();
        let inputs = inputs_of(&case.inputs);
        let cpu = case.build().eval(&inputs, &mut Outputs::new()).unwrap_err();
        let gpu = case
            .build()
            .eval_gpu(&ctx, &inputs, &mut Outputs::new())
            .unwrap_or_else(|| panic!("{label}: no GPU path"))
            .unwrap_err();
        assert_eq!(gpu, cpu, "{label}");
    }
}

#[test]
fn a_u8_blur_too_wide_for_32_bit_sums_stays_on_the_cpu() {
    let Some(ctx) = context() else { return };
    let mut rng = Rng(3);
    let f = Frame::from_data(2, 2, 1, FrameData::U8(vec![1, 2, 3, 4]));
    let blur = |r: &str| case("box_blur", &[("radius", r)], vec![("in", f.clone())]);
    let inputs = inputs_of(&blur("0").inputs);
    assert!(
        blur("2048")
            .build()
            .eval_gpu(&ctx, &inputs, &mut Outputs::new())
            .is_none()
    );
    assert!(
        blur("2047")
            .build()
            .eval_gpu(&ctx, &inputs, &mut Outputs::new())
            .is_some()
    );
    // f32 has no such limit.
    let g = rng.f32_frame(2, 2, 1);
    let inputs = inputs_of(&[("in", g)]);
    assert!(
        blur("2048")
            .build()
            .eval_gpu(&ctx, &inputs, &mut Outputs::new())
            .is_some()
    );
}

#[test]
fn every_gpu_capable_kind_has_cases() {
    let reg = builtin_registry();
    let covered: BTreeSet<&str> = cases().iter().map(|c| c.kind).collect();
    let mut capable = BTreeSet::new();
    for kind in reg.kinds() {
        // Defaults, plus values for the required params that have none.
        let mut params: Params = reg.defaults(kind);
        for (k, v) in [("w", "1"), ("h", "1"), ("channel", "red")] {
            params.entry(k.to_string()).or_insert_with(|| v.to_string());
        }
        let spec = NodeSpec {
            id: NodeId("n".into()),
            kind: kind.into(),
            params,
        };
        // Sources and sinks need real files to build; none of them has a
        // GPU path, and their kinds are listed so a new one is noticed.
        match reg.build(&spec) {
            Ok(node) if node.supports_gpu() => {
                capable.insert(kind);
            }
            Ok(_) => {}
            Err(e) => assert!(
                ["y4m_source", "y4m_sink", "ffmpeg_source", "ffmpeg_sink"].contains(&kind),
                "{kind} did not build from its defaults: {e}"
            ),
        }
    }
    let missing: Vec<_> = capable.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "GPU-capable kinds without conformance cases: {missing:?}"
    );
    assert_eq!(
        capable.into_iter().collect::<Vec<_>>(),
        [
            "box_blur",
            "cast",
            "crop",
            "gain",
            "grayscale",
            "invert",
            "merge",
            "split",
            "threshold"
        ]
    );
}
