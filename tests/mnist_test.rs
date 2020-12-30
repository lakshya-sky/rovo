use byteorder::{BigEndian, ReadBytesExt};
use rovo::{
    aten::native::{full, scalar_tensor, wrapped_scalar_tensor},
    autograd::{backward, tensor},
    c10::TensorOptions,
    core::manual_seed,
    init_rovo,
    nn::{Functional, Linear, Module, Sequential},
    optim::{Optimizer, SGDOptions, SGD},
    tensor::{
        log_softmax,
        loss::{binary_cross_entropy, Reduction},
        Tensor,
    },
};
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
        images.push(tensor(image_data.as_slice(), None).view(&[1, 784]));
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

    let mut model = Sequential::new();

    model.add(Linear::new(784, 64));
    model.add(Functional::new(Functional::sigmoid()));
    model.add(Linear::new(64, 10));
    model.add(Functional::new(Functional::sigmoid()));

    let mut sgd = SGD::new(model.parameters().unwrap(), SGDOptions::new(0.1));

    let step = |optimizer: &mut SGD, model: &Sequential, inputs: Tensor, target: Tensor| {
        // Note: Can't put the following line into closure beacuse
        // zero_grad uses immutable reference and step uses mutable reference.
        optimizer.zero_grad();
        let closure = || {
            let y = model.forward(&[&inputs]);
            let y = log_softmax(&y, 1, None);
            println!("LogSoftmax result: {:?}", y);
            let loss = binary_cross_entropy(&y, &target, None, Reduction::Mean);
            backward::backward(&vec![loss.clone()], &vec![], false);
            loss
        };
        optimizer.step(Some(closure))
    };

    let train_data = load_data("/Users/darshankathiriya/Downloads", "train").unwrap();
    for (index, data) in train_data.iter().enumerate() {
        let image = &data.image;
        let mut target = vec![0.0f32; 10];
        target[data.classification as usize] = 1.0;
        let target = tensor(target.as_slice(), None);
        let result = step(&mut sgd, &model, image.clone(), target);
        if index % 100 == 0 {
            println!("Loss: {:?}", result);
        }
    }
}
