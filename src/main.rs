use std::sync::Arc;

// --- 1. The Data Types --- 
//
// "Heavy" data is wrapped in Arc so it's cheap to pass around
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub buffer: Vec<u8>, //pixels: Vec<u8>, 
}

impl VideoFrame {
    // Helper fn
    pub fn len(&self) -> usize {
        (self.width * self.height * (self.channels as u32)) as usize
    }
    pub fn get_pixel(&self, x: u32, y: u32) -> &[u8] {
        let stride = self.width * (self.channels as u32);
        let start = (y * stride + x * (self.channels as u32)) as usize;
        let end = start + self.channels as usize;
        &self.buffer[start..end]
    }
}

// The unified enum
#[derive(Clone, Debug)]
pub enum Signal {
    // Heavy Data: Cheap to clone because it's just a pointer
    Image(Arc<VideoFrame>),
    // Light Control: Cheap to copy because it's just 8 bytes
    Value(f64),
    // Empty state for initialization
    Void,
}

// Helper to safely extract a float from a signal
impl Signal {
    fn as_float(&self) -> f64 {
        match self {
            Signal::Value(v) => *v,
            _ => 0.0,
        }
    }
    // Helper to get image or return empty
    fn as_image(&self) -> Option<&Arc<VideoFrame>> {
        match self {
            Signal::Image(img) => Some(img),
            _ => None,
        }
    }
}
// --- 2. The Logic Trait ---

trait NodeLogic {
    // We pass *all* inputs here. The Logic decides which is data and which is control.
    fn process(&mut self, inputs: &[Signal]) -> Signal;
}

// --- 3. Concrete Nodes ---

// A Generator Node (e.g., LFO)
struct SineWave {
    phase: f64,
}
impl NodeLogic for SineWave {
    fn process(&mut self, _inputs: &[Signal]) -> Signal {
        self.phase += 0.1;
        Signal::Value((self.phase.sin()+1.0)/2.0)
    }
}

// A. Source: Generates a solid color video frame
struct ColorSource {
    width: u32,
    height: u32,
}

impl NodeLogic for ColorSource {
    fn process(&mut self, inputs: &[Signal]) -> Signal {
        // Optional: Control color via Input[0] (0.0 to 1.0)
        let intensity = inputs.get(0).map(|s| s.as_float()).unwrap_or(1.0);
        
        // Create a dummy frame (Gray value = intensity * 255)
        let val = (intensity * 255.0) as u8;
        let size = (self.width * self.height * 3) as usize;
        let buffer = vec![val; size];

        Signal::Image(Arc::new(VideoFrame {
            width: self.width,
            height: self.height,
            channels: 3,
            buffer,
        }))
    }
}

// A Processor Node (Crop)
struct BrightnessEffect;
impl NodeLogic for BrightnessEffect {
    fn process(&mut self, inputs: &[Signal]) -> Signal {
        // Input 0: The Image
        // Input 1: The Brightness Factor (Control Signal)
        let factor = inputs.get(1).map(|s| s.as_float()).unwrap_or(1.0);
        if let Some(img_arc) = inputs.get(0).and_then(|s| s.as_image()) {
            // COPY-ON-WRITE:
            // We only allocate a new buffer because we are modifying data.
            // If we were just reading, we would pass the Arc through.
            let mut pixels = img_arc.buffer.clone();
            // Apply brightness (dummy logic)
            for p in pixels.iter_mut() {
                *p = (*p as f64 * factor).min(255.0) as u8;
            }

            return Signal::Image(Arc::new(VideoFrame {
                width: img_arc.width,
                height: img_arc.height,
                channels: img_arc.channels,
                buffer: pixels,
            }));
        }
        Signal::Void
    }
}

// C. Control: Low Frequency Oscillator (LFO)
struct LFO {
    phase: f64,
    speed: f64,
}

impl NodeLogic for LFO {
    fn process(&mut self, _inputs: &[Signal]) -> Signal {
        self.phase += self.speed;
        // Output a value between 0.0 and 1.0
        let val = (self.phase.sin() + 1.0) / 2.0;
        Signal::Value(val)
    }
}

// --- 4. The Graph Engine (Tick-Based) ---

struct NodeContainer {
    name: String,
    logic: Box<dyn NodeLogic>,
    // Indices of nodes we pull from
    inputs: Vec<usize>, 
    // Double buffer: [Read, Write]
    buffers: [Signal; 2],
    active_idx: usize,
}

impl NodeContainer {
    fn new(name: &str, logic: Box<dyn NodeLogic>, inputs: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            logic,
            inputs,
            buffers: [Signal::Void, Signal::Void],
            active_idx: 0,
        }
    }
}

struct Pipeline {
    nodes: Vec<NodeContainer>,
}

impl Pipeline {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn add_node(&mut self, name: &str, logic: Box<dyn NodeLogic>, inputs: Vec<usize>) -> usize {
        let node = NodeContainer::new(name, logic, inputs);
        self.nodes.push(node);
        self.nodes.len() - 1 // Return ID
    }

    fn step(&mut self) {
        println!("--- Tick ---");
        // --- STEP 1: SNAPSHOT ---
        // Create a read-only list of everyone's CURRENT output.
        // This decouples "reading the world" from "updating the nodes".
        // Cloning 'Signal' is cheap because it uses Arc.
        let world_state: Vec<Signal> = self.nodes
            .iter()
            .map(|n| n.buffers[n.active_idx].clone())
            .collect();

        // --- STEP 2: COMPUTE ---
        // Now we can iterate mutably without looking at 'self.nodes' again.
        // We look at 'world_state' instead.
        let mut next_values = Vec::new();
        for node in &mut self.nodes {
            // Gather inputs from the *previous* tick of upstream nodes
            let input_signals: Vec<Signal> = node.inputs
                .iter()
                .map(|&src_idx| {
                    world_state.get(src_idx).cloned().unwrap_or(Signal::Void)
                })
                .collect();

            // Run logic
            let result = node.logic.process(&input_signals);
            next_values.push(result);
        }

        // --- STEP 3. COMMIT / SWAP PHASE
        for (i, val) in next_values.into_iter().enumerate() {
            let node = &mut self.nodes[i];
            
            // Debug print to prove it works
            match &val {
                Signal::Value(v) => println!("  Node '{}' produced Value: {:.2}", node.name, v),
                Signal::Image(img) => println!("  Node '{}' produced Image: {}x{} [Byte 0: {}]", 
                    node.name, img.width, img.height, img.buffer.first().unwrap_or(&0)),
                Signal::Void => println!("  Node '{}' produced Void", node.name),
            }

            let write_idx = 1 - node.active_idx;
            node.buffers[write_idx] = val; // Write to future
            node.active_idx = write_idx;   // Flip "Future" to "Present"
        }
    }
}


fn main() {
    // do something
    println!("Hello, world!");

    let mut pipe = Pipeline::new();

    // Node 0: LFO (Control Signal)
    let lfo_id = pipe.add_node("LFO", Box::new(LFO { phase: 0.0, speed: 0.5 }), vec![]);

    // Node 1: Color Source (Data) - Controlled by LFO (Input 0)
    let src_id = pipe.add_node("Source", Box::new(ColorSource { width: 2, height: 2 }), vec![lfo_id]);

    // Node 2: Brightness Effect - Takes Source (Input 0) and LFO (Input 1)
    let bright_id = pipe.add_node("Brightness", Box::new(BrightnessEffect), vec![src_id, lfo_id]);

    // Run 5 frames
    for _ in 0..5 {
        pipe.step();
    }
}
