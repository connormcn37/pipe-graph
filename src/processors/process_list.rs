use crate::{data::Frame, traits::Processor};

#[derive(Default)]
pub struct ProcessList {
    pub processes: Vec<Box<dyn Processor>>,
}

impl ProcessList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_processor<P: Processor + 'static>(&mut self, processor: P) {
        self.processes.push(Box::new(processor));
    }
}

impl Processor for ProcessList {
    fn process(&self, input: &mut Frame) {
        for processor in &self.processes {
            processor.process(input);
        }
    }
}