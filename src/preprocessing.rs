use image::GenericImageView;
use image::imageops::FilterType;
use ndarray::{Ix4, OwnedRepr};

pub struct Image {
    pub original_width: u32,
    pub original_height: u32,
    pub tensor: ndarray:: ArrayBase<OwnedRepr<f32>, Ix4>
}

impl Image {
    pub fn new(bytes: &[u8],original_width: u32, original_height: u32) -> Result<Self, image::ImageError> {
        let original_image = image::load_from_memory(bytes)?;
        let sized_image = original_image.resize_exact(640, 640,FilterType::Gaussian );
        let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

        for pixel in sized_image.pixels(){
            let (x, y, colors) = pixel;
            tensor[[0, 0, y as usize, x as usize]] = (colors[0] as f32) / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = (colors[1] as f32) / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = (colors[2] as f32) / 255.0;
        }
        return Ok(Self{
            original_width,
            original_height,
            tensor
        })
    }
}
