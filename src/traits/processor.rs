use crate::data::Frame;

/// A single-in/single-out in-place frame transform.
///
/// `Send` so a [`crate::exec::ProcessorNode`] wrapping it can be evaluated on a
/// worker thread under [`crate::exec::ExecMode::Parallel`].
pub trait Processor: Send {
    fn process(&self, input: &mut Frame);
}
