use std::path::PathBuf;

use crate::data::{Frame, FrameData, Payload, PayloadKind};
use crate::exec::{BuildError, Inputs, Node, NodeError, Outputs, ParamsExt, PortSet, PortSpec};
use crate::graph::Params;
use image::ColorType;

/// Stage that reads an image from a file path.
pub struct ImageReadStage {
    path: PathBuf,
}

impl ImageReadStage {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl TryFrom<&Params> for ImageReadStage {
    type Error = BuildError;

    fn try_from(p: &Params) -> Result<Self, BuildError> {
        Ok(Self {
            path: PathBuf::from(p.get_str("path")?),
        })
    }
}

impl Node for ImageReadStage {
    fn ports(&self) -> PortSet {
        PortSet::new(
            vec![],
            vec![PortSpec::new("out", PayloadKind::Frame)],
        )
    }

    fn eval(&mut self, _inputs: &Inputs, outputs: &mut Outputs) -> Result<(), NodeError> {
        let img = image::open(&self.path)
            .map_err(|e| NodeError::Message(format!("failed to open image: {}", e)))?
            .to_rgb8();

        let (w, h) = img.dimensions();
        let frame = Frame::from_data(
            w,
            h,
            3,
            FrameData::U8(img.into_raw()),
        );

        outputs.set("out", Payload::Frame(frame));
        Ok(())
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
        PortSet::new(
            vec![PortSpec::new("in", PayloadKind::Frame)],
            vec![],
        )
    }

    fn eval(&mut self, inputs: &Inputs, _outputs: &mut Outputs) -> Result<(), NodeError> {
        let frame = inputs.frame("in")?;

        let u8_data = frame.as_u8().ok_or_else(|| {
            NodeError::Message("ImageWriteStage requires a u8 frame".to_string())
        })?;

        if frame.channels == 3 {
            image::save_buffer(
                &self.path,
                u8_data,
                frame.width,
                frame.height,
                ColorType::Rgb8,
            )
            .map_err(|e| NodeError::Message(format!("failed to save image: {}", e)))?;
        } else if frame.channels == 4 {
            image::save_buffer(
                &self.path,
                u8_data,
                frame.width,
                frame.height,
                ColorType::Rgba8,
            )
            .map_err(|e| NodeError::Message(format!("failed to save image: {}", e)))?;
        } else if frame.channels == 1 {
            image::save_buffer(
                &self.path,
                u8_data,
                frame.width,
                frame.height,
                ColorType::L8,
            )
            .map_err(|e| NodeError::Message(format!("failed to save image: {}", e)))?;
        } else {
            return Err(NodeError::Message(format!(
                "unsupported channel count for writing: {}",
                frame.channels
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        m.insert(crate::graph::PortId("in".to_string()), Payload::Frame(f.clone()));
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
}
