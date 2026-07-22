use std::path::Path;
use ndarray::{Array, IxDyn};
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use crate::preprocessing::Image;

struct Detector {
    session : Session
}

impl Detector {
    pub fn new(path_to_model: &Path, cuda: bool, device_id: i32) -> Result<Self, ort::Error>{
       if (cuda == true) {
           let session = Session::builder()?.with_execution_providers([ep::CUDA::default().with_device_id(device_id).build()])?.commit_from_file(path_to_model)?;
           Ok(Self{
               session
           })
       }else{
           let session = Session::builder()?.commit_from_file(path_to_model)?;
           Ok(Self{session})
       }
    }

    pub fn process(&mut self, image: &Image) -> Result<Array<f32, IxDyn>, ort::Error>{
        let tensor = TensorRef::from_array_view(&image.tensor)?;
       let outputs = self.session.run(ort::inputs![tensor])?;
        return Ok(outputs["output0"].try_extract_array::<f32>()?.to_owned());
    }

}