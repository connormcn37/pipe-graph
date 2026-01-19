use crate::data::Frame;

pub trait Processor {
    fn process(&self, input: &mut Frame);
}