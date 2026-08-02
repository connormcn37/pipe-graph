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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::{Channel, ClearChannel};

    /// Golden test mirroring the `main.rs` demo: a nested `ProcessList` that
    /// clears red + green (inner) then blue (outer) must zero every channel.
    /// This locks in the numeric result so the Phase 1 `Frame` refactor can't
    /// silently change behavior.
    #[test]
    fn nested_demo_clears_all_channels() {
        let mut frame = Frame {
            width: 3,
            height: 3,
            pixels: vec![
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (0, 255, 255),
                (255, 0, 255),
                (192, 192, 192),
                (128, 128, 128),
                (64, 64, 64),
            ],
        };

        let mut inner = ProcessList::new();
        inner.add_processor(ClearChannel(Channel::Red));
        inner.add_processor(ClearChannel(Channel::Green));

        let mut outer = ProcessList::new();
        outer.add_processor(inner);
        outer.add_processor(ClearChannel(Channel::Blue));

        outer.process(&mut frame);

        assert!(frame.pixels.iter().all(|&p| p == (0, 0, 0)));
    }

    #[test]
    fn empty_list_is_identity() {
        let mut frame = Frame {
            width: 1,
            height: 1,
            pixels: vec![(10, 20, 30)],
        };
        ProcessList::new().process(&mut frame);
        assert_eq!(frame.pixels, vec![(10, 20, 30)]);
    }
}
