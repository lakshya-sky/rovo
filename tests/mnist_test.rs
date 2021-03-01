use byteorder::{BigEndian, ReadBytesExt};
use rovo::{
    aten::native::argmax,
    autograd::{backward, tensor},
    core::manual_seed,
    init_rovo,
    nn::{nll_loss, Functional, Linear, Module, NLLLossFuncOptions, Sequential},
    optim::{Optimizer, SGDOptions, SGDOptionsBuilder, SGD},
    tensor::{log_softmax, Tensor},
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
pub fn load_data(
    dataset_dir: &str,
    dataset_name: &str,
    limit: usize,
) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}/{}-labels-idx1-ubyte.gz", dataset_dir, dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}/{}-images-idx3-ubyte.gz", dataset_dir, dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Tensor> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;
    let mut count = images_data.sizes[0] as usize;
    if limit > 0 && limit < (count) {
        count = limit;
    }
    for i in 0..count {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f32> = image_data.into_iter().map(|x| x as f32 / 255.).collect();
        let tensor = tensor(image_data.as_slice(), None);
        let tensor = tensor.view(&[1, 784]);
        images.push(tensor);
    }
    let classifications: Vec<u8> = label_data.data.clone();
    let mut ret: Vec<MnistImage> = Vec::new();
    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        });
    }
    Ok(ret)
}
#[test]
fn mnist_nn() {
    init_rovo();
    manual_seed(1);
    let mut model = Sequential::new();
    model.add(Linear::new(784, 32));
    model.add(Functional::new(Functional::sigmoid()));
    model.add(Linear::new(32, 10));
    let sgd_options = SGDOptionsBuilder::new(0.01).momentum(0.0).build();
    let mut sgd = SGD::new(model.parameters().unwrap(), sgd_options);
    let step = |optimizer: &mut SGD, model: &Sequential, inputs: Tensor, target: Tensor| {
        optimizer.zero_grad();
        let closure = || {
            let y = model.forward(&[&inputs]);
            let y = log_softmax(&y, 1, None);
            let loss = nll_loss(&y, &target, NLLLossFuncOptions::default());
            backward::backward(&vec![loss.clone()], &vec![], false);
            loss
        };
        optimizer.step(Some(closure))
    };
    let train_data = load_data("/Users/darshankathiriya/Downloads", "train", 20000).unwrap();
    let test_data = load_data("/Users/darshankathiriya/Downloads", "t10k", 1000).unwrap();
    let mut test_iter = test_data.iter();
    for (index, data) in train_data.iter().enumerate() {
        let image = &data.image;
        let target = tensor(data.classification as i64, None);
        target.resize(&[1], None);
        let result = step(&mut sgd, &model, image.clone(), target);
        if index % 1000 == 0 {
            println!("Loss: {:?}", result,);
            let test_item = test_iter.next().unwrap();
            let test_x = &test_item.image;
            let test_y = test_item.classification;
            let y_hat = model.forward(&[test_x]);
            let y_hat = log_softmax(&y_hat, 1, None);
            println!(
                "Test Result: target= {:?}, prediction= {:?}",
                test_y,
                argmax(&y_hat, None, false)
            );
        }
    }
}
