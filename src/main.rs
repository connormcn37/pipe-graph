use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;

// --- Data Structures ---

#[derive(Clone, Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub data: Vec<u8>, // Simplified data buffer
}

impl Frame {
    fn new(w: u32, h: u32, c: u32) -> Self {
        Frame { width: w, height: h, channels: c, data: vec![0; (w * h * c) as usize] }
    }
}

// Flexible parameter types
#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i32),
    Float(f64),
    Bool(bool),
}

// --- The Base Components (Composition over Inheritance) ---

// The "Base Class" data that every entity has
#[derive(Debug)]
struct BaseEntity {
    label: String,
    inputs: Vec<String>, // Labels of input entities
}

impl BaseEntity {
    fn new(label: &str) -> Self {
        Self { label: label.to_string(), inputs: Vec::new() }
    }
}

// --- Traits (Interfaces) ---

pub trait Entity {
    fn label(&self) -> &str;
    fn get_input_labels(&self) -> &[String];
    fn connect(&mut self, input_labels: Vec<String>);
    fn disconnect(&mut self);
}

pub trait Stage: Entity + fmt::Debug {
    fn set_parameter(&mut self, key: &str, value: ParamValue);
    
    // Returns the cached output of this stage
    fn get_last_frame(&self) -> Option<Frame>;
    
    // The core logic: takes resolved input frames, runs logic, updates internal state
    fn process(&mut self, inputs: Vec<&Frame>) -> Result<(), String>;
}

// --- Concrete Stage Implementations ---

#[derive(Debug)]
struct CropStage {
    base: BaseEntity,
    params: HashMap<String, ParamValue>,
    last_output: Option<Frame>,
}

impl CropStage {
    fn new(label: &str) -> Self {
        Self {
            base: BaseEntity::new(label),
            params: HashMap::new(),
            last_output: None,
        }
    }
}

impl Entity for CropStage {
    fn label(&self) -> &str { &self.base.label }
    fn get_input_labels(&self) -> &[String] { &self.base.inputs }
    fn connect(&mut self, inputs: Vec<String>) { self.base.inputs = inputs; }
    fn disconnect(&mut self) { self.base.inputs.clear(); self.last_output = None; }
}

impl Stage for CropStage {
    fn set_parameter(&mut self, key: &str, value: ParamValue) {
        self.params.insert(key.to_string(), value);
    }

    fn get_last_frame(&self) -> Option<Frame> {
        self.last_output.clone()
    }

    fn process(&mut self, inputs: Vec<&Frame>) -> Result<(), String> {
        if inputs.is_empty() { return Err("CropStage requires 1 input".into()); }
        
        let input = inputs[0];
        // logic: Perform simulated crop (logic simplified for brevity)
        // In a real app, read self.params["x"], self.params["width"], etc.
        println!("  -> [{}] Cropping frame of size {}x{}", self.label(), input.width, input.height);
        
        // Save result
        self.last_output = Some(Frame::new(100, 100, input.channels)); 
        Ok(())
    }
}

#[derive(Debug)]
struct MergeStage {
    base: BaseEntity,
    last_output: Option<Frame>,
}

impl MergeStage {
    fn new(label: &str) -> Self {
        Self { base: BaseEntity::new(label), last_output: None }
    }
}

impl Entity for MergeStage {
    fn label(&self) -> &str { &self.base.label }
    fn get_input_labels(&self) -> &[String] { &self.base.inputs }
    fn connect(&mut self, inputs: Vec<String>) { self.base.inputs = inputs; }
    fn disconnect(&mut self) { self.base.inputs.clear(); self.last_output = None; }
}

impl Stage for MergeStage {
    fn set_parameter(&mut self, _key: &str, _value: ParamValue) {} // No params needed

    fn get_last_frame(&self) -> Option<Frame> { self.last_output.clone() }

    fn process(&mut self, inputs: Vec<&Frame>) -> Result<(), String> {
        if inputs.len() < 2 { return Err("MergeStage requires at least 2 inputs".into()); }
        
        println!("  -> [{}] Merging {} inputs...", self.label(), inputs.len());
        
        // Logic: Create a frame with K channels
        let k = inputs.len() as u32;
        self.last_output = Some(Frame::new(inputs[0].width, inputs[0].height, k));
        Ok(())
    }
}

// Source Stage (Mock Camera/File Reader)
#[derive(Debug)]
struct SourceStage {
    base: BaseEntity,
    last_output: Option<Frame>,
}
impl SourceStage {
    fn new(label: &str) -> Self { Self { base: BaseEntity::new(label), last_output: None } }
}
impl Entity for SourceStage {
    fn label(&self) -> &str { &self.base.label }
    fn get_input_labels(&self) -> &[String] { &[] } // Source has no inputs
    fn connect(&mut self, _inputs: Vec<String>) {}
    fn disconnect(&mut self) {}
}
impl Stage for SourceStage {
    fn set_parameter(&mut self, _key: &str, _value: ParamValue) {}
    fn get_last_frame(&self) -> Option<Frame> { self.last_output.clone() }
    fn process(&mut self, _inputs: Vec<&Frame>) -> Result<(), String> {
        println!("  -> [{}] Generating new frame", self.label());
        self.last_output = Some(Frame::new(1920, 1080, 3));
        Ok(())
    }
}

// --- The Pipeline ---

pub struct Pipeline {
    // We use Rc<RefCell<dyn Stage>> to allow shared mutable ownership
    // The HashMap ensures unique labels
    stages: HashMap<String, Rc<RefCell<dyn Stage>>>,
    execution_order: Vec<String>, // Simple list for sequential execution
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    pub fn add_stage(&mut self, stage: Box<dyn Stage>) -> Result<(), String> {
        let label = stage.label().to_string();
        if self.stages.contains_key(&label) {
            return Err(format!("Stage with label '{}' already exists", label));
        }
        
        // Add to map
        self.stages.insert(label.clone(), Rc::new(stage));
        // Add to execution order (in a real app, you'd calculate topological sort)
        self.execution_order.push(label);
        Ok(())
    }

    pub fn connect(&mut self, from_label: &str, to_label: &str) -> Result<(), String> {
        // 1. Get the target stage
        let target_rc = self.stages.get(to_label)
            .ok_or(format!("Target stage {} not found", to_label))?;
            
        // 2. Validate source exists
        if !self.stages.contains_key(from_label) {
            return Err(format!("Source stage {} not found", from_label));
        }

        // 3. Update the target's inputs
        // Note: In a real implementation, you might want to append, not overwrite.
        // For now, we fetch existing inputs and add the new one.
        let mut target = target_rc.borrow_mut();
        let mut inputs = target.get_input_labels().to_vec();
        inputs.push(from_label.to_string());
        target.connect(inputs);
        
        Ok(())
    }

    pub fn step(&mut self) {
        println!("--- Pipeline Step ---");
        
        // We iterate through our defined execution order
        for label in &self.execution_order {
            // We need to borrow the stage to run it
            // We must drop the borrow before moving to the next iteration if possible, 
            // though here we hold it only for the duration of resolving inputs and processing.
            
            let stage_rc = self.stages.get(label).unwrap();
            let input_labels = stage_rc.borrow().get_input_labels().to_vec();
            
            // Resolve Inputs:
            // We need to look up the frames from previous stages.
            let mut input_frames = Vec::new();
            let mut valid_inputs = true;

            // Gather inputs from the HashMap
            for input_label in input_labels {
                if let Some(parent_rc) = self.stages.get(&input_label) {
                    let parent = parent_rc.borrow();
                    if let Some(frame) = parent.get_last_frame() {
                        // We clone the frame (expensive) or the frame needs to be wrapped in Rc
                        // For this demo, we assume Frame is lightweight or we accept the clone
                        input_frames.push(frame); 
                    } else {
                        println!("Warning: Parent {} has no output yet.", input_label);
                        valid_inputs = false;
                    }
                }
            }

            if valid_inputs {
                // To pass references of frames to `process`, we need a temporary vector of references
                // This is a bit tricky in Rust due to lifetimes, so we do it in a tight scope.
                 let _ = stage_rc.borrow_mut().process(input_frames.iter().collect());
            }
        }
    }
}

// --- Usage Example ---

fn main() {
    let mut pipeline = Pipeline::new();

    // 1. Create Stages
    let source1 = SourceStage::new("cam_1");
    let source2 = SourceStage::new("cam_2");
    let cropper = CropStage::new("crop_1");
    let merger = MergeStage::new("merge_final");

    // 2. Add to Pipeline
    pipeline.add_stage(Box::new(source1)).unwrap();
    pipeline.add_stage(Box::new(source2)).unwrap();
    pipeline.add_stage(Box::new(cropper)).unwrap();
    pipeline.add_stage(Box::new(merger)).unwrap();

    // 3. Connect (build the graph)
    // cam_1 -> crop_1
    pipeline.connect("cam_1", "crop_1").unwrap();
    
    // crop_1 -> merge_final
    pipeline.connect("crop_1", "merge_final").unwrap();
    
    // cam_2 -> merge_final
    pipeline.connect("cam_2", "merge_final").unwrap();

    // 4. Run loop
    for _ in 0..3 {
        pipeline.step();
    }
}
