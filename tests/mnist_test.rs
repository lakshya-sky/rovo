use byteorder::{BigEndian, ReadBytesExt};
use rovo::{autograd::tensor, core::manual_seed, init_rovo, tensor::Tensor};
use std::{
    fs::File,
    io::{Cursor, Read},
};

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = flate2::read::GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

#[derive(Debug)]
pub struct MnistImage {
    pub image: Tensor,
    pub classification: u8,
}
pub fn load_data(dataset_dir: &str, dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}/{}-labels-idx1-ubyte.gz", dataset_dir, dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}/{}-images-idx3-ubyte.gz", dataset_dir, dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Tensor> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f32> = image_data.into_iter().map(|x| x as f32 / 255.).collect();
        images.push(tensor(image_data.as_slice(), None).view(&[1,784]));
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        })
    }

    Ok(ret)
}

#[test]
fn mnist_nn() {
    init_rovo();
    manual_seed(1);
    let train_data = load_data("/home/darshan/Downloads", "train").unwrap();
    println!("{:?}", train_data[0]);
}
