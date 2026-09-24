use std::path::PathBuf;
use std::sync::Arc;

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{
    BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec, Registry,
};
use crate::graph::Params;
use image::ColorType;

/// Stage that reads an image from a file path.
///
/// The file is decoded once and the same buffer is re-published on every
/// evaluation (see [`Outputs::set_shared`]), so a still image feeding a stream
/// costs one decode and consumers can detect "unchanged" by pointer identity.
/// [`Node::reset`] drops the cache so the next evaluation re-reads the file.
pub struct ImageReadStage {
    path: PathBuf,
    cached: Option<Arc<Payload>>,
}

impl ImageReadStage {
    pub fn new(path: PathBuf) -> Self {
        Self { path, cached: None }
    }
}

impl TryFrom<&Params> for ImageReadStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self::new(PathBuf::from(p.get_str("path")?)))
    }
}

impl Node for ImageReadStage {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![], vec![PortSpec::new("out", PayloadKind::Frame)])
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let payload = match &self.cached {
            Some(p) => p.clone(),
            None => {
                let img = image::open(&self.path)
                    .map_err(|e| NodeError::Message(format!("failed to open image: {e}")))?
                    .to_rgb8();
                let (w, h) = img.dimensions();
                let frame = Frame::from_data(w, h, 3, FrameData::U8(img.into_raw()));
                self.cached.insert(Arc::new(Payload::Frame(frame))).clone()
            }
        };
        outputs.set_shared("out", payload);
        Ok(())
    }

    fn reset(&mut self) {
        self.cached = None;
    }
}

/// Stage that writes an input frame to a file path.
pub struct ImageWriteStage {
    path: PathBuf,
}

impl ImageWriteStage {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl TryFrom<&Params> for ImageWriteStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self {
            path: PathBuf::from(p.get_str("path")?),
        })
    }
}

impl Node for ImageWriteStage {
    fn ports(&self) -> PortSet {
        PortSet::new(vec![PortSpec::new("in", PayloadKind::Frame)], vec![])
    }

    fn eval(&mut self, inputs: &Inputs, _outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;

        let u8_data = frame
            .as_u8()
            .ok_or_else(|| NodeError::Message("ImageWriteStage requires a u8 frame".to_string()))?;

        let color = match frame.channels {
            1 => ColorType::L8,
            3 => ColorType::Rgb8,
            4 => ColorType::Rgba8,
            n => {
                return Err(NodeError::Message(format!(
                    "unsupported channel count for writing: {n}"
                )));
            }
        };
        image::save_buffer(&self.path, u8_data, frame.width, frame.height, color)
            .map_err(|e| NodeError::Message(format!("failed to save image: {e}")))?;

        Ok(())
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_stage::<ImageReadStage>("image_read");
    reg.register_stage::<ImageWriteStage>("image_write");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn write_then_read_round_trips() {
        let path = std::env::temp_dir().join("test_pipe_graph_io.png");

        let f = Frame::from_rgb8(
            2,
            2,
            vec![(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30)],
        );

        let mut write_stage = ImageWriteStage::new(path.clone());
        let mut m = std::collections::HashMap::new();
        m.insert(
            crate::graph::PortId("in".to_string()),
            Payload::Frame(f.clone()),
        );
        let mut write_out = Outputs::new();
        write_stage.eval(&Inputs::new(m), &mut write_out).unwrap();

        let mut read_stage = ImageReadStage::new(path.clone());
        let mut read_out = Outputs::new();
        read_stage.eval(&Inputs::default(), &mut read_out).unwrap();

        let loaded = read_out.get("out").unwrap().as_frame().unwrap();
        assert_eq!(loaded.width, 2);
        assert_eq!(loaded.height, 2);
        assert_eq!(loaded.channels, 3);
        assert_eq!(loaded.to_rgb8(), f.to_rgb8());

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_caches_decoded_image_until_reset() {
        let path = std::env::temp_dir().join("test_pipe_graph_read_cache.png");
        image::save_buffer(&path, &[1, 2, 3], 1, 1, ColorType::Rgb8).unwrap();

        let mut stage = ImageReadStage::new(path.clone());
        let mut first = Outputs::new();
        stage.eval(&Inputs::default(), &mut first).unwrap();

        // Change the file on disk: a cached read must not notice.
        image::save_buffer(&path, &[9, 9, 9], 1, 1, ColorType::Rgb8).unwrap();
        let mut second = Outputs::new();
        stage.eval(&Inputs::default(), &mut second).unwrap();
        assert!(Arc::ptr_eq(
            &first.get_shared("out").unwrap(),
            &second.get_shared("out").unwrap()
        ));

        // After a reset the file is decoded again.
        stage.reset();
        let mut third = Outputs::new();
        stage.eval(&Inputs::default(), &mut third).unwrap();
        let f = third.get("out").unwrap().as_frame().unwrap();
        assert_eq!(f.to_rgb8(), vec![(9, 9, 9)]);

        let _ = std::fs::remove_file(path);
    }
}
