use std::fs::File;
use std::io::{prelude::*, Error, ErrorKind};
use std::iter::zip;
use std::{io, vec};

const IMAGE_SIZE: usize = 28;
const HEADER_IMAGE_MAGIC: u32 = 2051;
const HEADER_LABEL_MAGIC: u32 = 2049;
const NUM_TRAIN_IMAGES: usize = 60000;
const NUM_EVAL_IMAGES: usize = 10000;

struct Image {
    pixels: [f32; IMAGE_SIZE * IMAGE_SIZE],
}

fn load_mnist_labels(file_path: &str, n: usize) -> Result<Vec<u8>, Error> {
    let label_file = File::open(file_path)?;
    let label_file = io::BufReader::new(label_file);

    let mut buffer_u32 = [0; 4];

    label_file.get_ref().read_exact(&mut buffer_u32)?;
    let magic_number = u32::from_be_bytes(buffer_u32);
    if magic_number != HEADER_LABEL_MAGIC {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid magic number for label file",
        ));
    }

    label_file.get_ref().read_exact(&mut buffer_u32)?;
    let num_labels = u32::from_be_bytes(buffer_u32);
    if num_labels != n as u32 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid number of labels",
        ));
    }

    let mut labels = vec![0; n];
    label_file.get_ref().read_exact(labels.as_mut_slice())?;

    Ok(labels)
}

fn load_mnist_images(file_path: &str, n: usize) -> Result<Vec<Image>, Error> {
    let image_file = File::open(file_path)?;
    let image_file = io::BufReader::new(image_file);

    let mut buffer_u32 = [0; 4];

    image_file.get_ref().read_exact(&mut buffer_u32)?;
    let magic_number = u32::from_be_bytes(buffer_u32);
    if magic_number != HEADER_IMAGE_MAGIC {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid magic number for image file",
        ));
    }

    image_file.get_ref().read_exact(&mut buffer_u32)?;
    let num_images = u32::from_be_bytes(buffer_u32);
    if num_images != n as u32 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid number of images",
        ));
    }

    image_file.get_ref().read_exact(&mut buffer_u32)?;
    let rows = u32::from_be_bytes(buffer_u32);
    if rows != IMAGE_SIZE as u32 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid number of rows"));
    }

    image_file.get_ref().read_exact(&mut buffer_u32)?;
    let columns = u32::from_be_bytes(buffer_u32);
    if columns != IMAGE_SIZE as u32 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid number of columns",
        ));
    }

    let mut images = Vec::with_capacity(n);
    for _ in 0..num_images {
        let mut buffer = [0; IMAGE_SIZE * IMAGE_SIZE];
        image_file
            .get_ref()
            .take((IMAGE_SIZE * IMAGE_SIZE) as u64)
            .read_exact(&mut buffer)?;
        let pixels: Vec<f32> = buffer.iter().map(|&x| x as f32 / 255.0).collect();
        images.push(Image {
            pixels: pixels.as_slice().try_into().unwrap(),
        });
    }

    Ok(images)
}

fn debug_print_image(image: &Image) {
    for i in 0..IMAGE_SIZE {
        for j in 0..IMAGE_SIZE {
            let val = image.pixels[i * IMAGE_SIZE + j];
            if val < 0.05 {
                print!(" ");
            } else if val < 0.5 {
                print!("*");
            } else {
                print!("#");
            }
        }
        println!();
    }
}

pub struct RandomSeries {
    state: u32,
}

impl RandomSeries {
    pub fn new(seed: u32) -> Self {
        RandomSeries { state: seed }
    }

    fn xorshift32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    pub fn rand_u32(&mut self, min: u32, max: u32) -> u32 {
        self.xorshift32() % (max - min) + min
    }

    pub fn rand(&mut self) -> f32 {
        //(self.xorshift32() as f64 / u32::MAX as f64) as f32
        f32::from_le_bytes((self.xorshift32() & 0x007fffff | 0x3f800000).to_le_bytes()) - 1.0
    }

    pub fn randn(&mut self) -> f32 {
        //-1.0f + self.rand() * 2.0f
        f32::from_le_bytes((self.xorshift32() & 0x007fffff | 0x40000000).to_le_bytes()) - 3.0
    }

    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        self.rand() * (max - min) + min
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-z))
}

fn sigmoid_prime(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn sigmoid_v(z: &[f32]) -> Vec<f32> {
    z.iter().map(|z| sigmoid(*z)).collect()
}

fn sigmoid_prime_v(z: &[f32]) -> Vec<f32> {
    z.iter().map(|z| sigmoid_prime(*z)).collect()
}

fn sum_v(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
}

fn hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
}

fn matrix_vec_mul(matrix: &[f32], vector: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; matrix.len() / vector.len()];
    for i in 0..result.len() {
        for j in 0..vector.len() {
            result[i] += matrix[i * vector.len() + j] * vector[j];
        }
    }
    result
}

fn dot_col_row(v_column: &[f32], v_row: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; v_column.len() * v_row.len()];
    for i in 0..v_column.len() {
        for j in 0..v_row.len() {
            result[i * v_row.len() + j] = v_column[i] * v_row[j];
        }
    }
    result
}

fn matrix_transpose(a: &[f32], matrix_width: usize) -> Vec<f32> {
    let mut result = vec![0.0; a.len()];
    for i in 0..matrix_width {
        for j in 0..matrix_width {
            result[i * matrix_width + j] = a[j * matrix_width + i];
        }
    }
    result
}

struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

struct Network {
    layers: Vec<Layer>,
    series: RandomSeries,
}

impl Network {
    fn new(seed: u32, layer_sizes: &[usize]) -> Network {
        let mut r = RandomSeries::new(seed);
        Network {
            layers: layer_sizes
                .iter()
                .zip(layer_sizes.iter().skip(1))
                .map(|(n_in, n_out)| Layer {
                    weights: (0..*n_in * *n_out).map(|_| r.randn() * r.rand()).collect(),
                    biases: (0..*n_out).map(|_| r.randn()).collect(),
                })
                .collect(),
            series: r,
        }
    }

    fn feedforward(&mut self, image: &Image) -> Vec<f32> {
        let mut activation = image.pixels.to_vec();
        for layer in &self.layers {
            activation = sigmoid_v(&sum_v(
                &matrix_vec_mul(&layer.weights, &activation),
                &layer.biases,
            ))
        }
        activation
    }

    fn predict_label(&mut self, image: &Image) -> u8 {
        let output = self.feedforward(image);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u8
    }

    fn backprop(&mut self, image: &Image, label: u8) -> Vec<Layer> {
        // feedforward
        let mut activation = image.pixels.to_vec();
        let mut activations = vec![activation.clone()];
        let mut weighted_inputs = vec![];

        for layer in self.layers.iter() {
            let z = sum_v(&matrix_vec_mul(&layer.weights, &activation), &layer.biases);
            activation = sigmoid_v(&z);
            weighted_inputs.push(z);
            activations.push(activation.clone());
        }

        // back propagation
        let mut nabla_layers: Vec<Layer> = self
            .layers
            .iter()
            .map(|layer| Layer {
                weights: vec![0.0; layer.weights.len()],
                biases: vec![0.0; layer.biases.len()],
            })
            .collect();

        let final_activation = activations.last().unwrap();
        let mut error = hadamard(
            &self.cost_derivative(final_activation, label),
            &sigmoid_prime_v(weighted_inputs.last().unwrap()),
        );

        let len = self.layers.len();
        nabla_layers[len - 1].biases = error.clone();
        nabla_layers[len - 1].weights = dot_col_row(&error, &activations[activations.len() - 2]);

        for l in 2..self.layers.len() + 1 {
            let z = &weighted_inputs[len - l];
            error = hadamard(
                &matrix_vec_mul(
                    &matrix_transpose(
                        &self.layers[len - l + 1].weights,
                        self.layers[len - l + 1].biases.len(),
                    ),
                    &error,
                ),
                &sigmoid_prime_v(z),
            );

            nabla_layers[len - l].biases = error.clone();
            nabla_layers[len - l].weights =
                dot_col_row(&error, &activations[activations.len() - l - 1]);
        }

        nabla_layers
    }

    fn update_mini_batch(
        &mut self,
        train_data: &(Vec<Image>, Vec<u8>),
        batch_start_index: usize,
        batch_end_index: usize,
        eta: f32,
        lmbda: f32,
    ) {
        let mut nabla_layers: Vec<Layer> = self
            .layers
            .iter()
            .map(|layer| Layer {
                weights: vec![0.0; layer.weights.len()],
                biases: vec![0.0; layer.biases.len()],
            })
            .collect();

        for i in batch_start_index..batch_end_index {
            let image = &train_data.0[i];
            let label = train_data.1[i];
            let delta_nabla = self.backprop(&image, label);
            for (nabla_layer, delta_nabla) in zip(nabla_layers.iter_mut(), delta_nabla.iter()) {
                nabla_layer.weights = sum_v(&nabla_layer.weights, &delta_nabla.weights);
                nabla_layer.biases = sum_v(&nabla_layer.biases, &delta_nabla.biases);
            }
        }

        let weight = eta / (batch_end_index - batch_start_index) as f32;
        let n = train_data.0.len();
        let regularization = 1.0 - eta * (lmbda / (n as f32));
        for (layer, nabla_layer) in zip(self.layers.iter_mut(), nabla_layers.iter()) {
            layer.weights = layer
                .weights
                .iter()
                .zip(nabla_layer.weights.iter())
                .map(|(w, nw)| regularization * w - weight * nw)
                .collect();
            layer.biases = layer
                .biases
                .iter()
                .zip(nabla_layer.biases.iter())
                .map(|(b, nb)| b - weight * nb)
                .collect();
        }
    }

    fn train_sgd(
        &mut self,
        train_data: &(Vec<Image>, Vec<u8>),
        eval_data: &(Vec<Image>, Vec<u8>),
        epochs: usize,
        batch_size: usize,
        eta: f32,
        lmbda: f32,
    ) {
        for epoch in 0..epochs {
            let learning_rate = (1.0 - (epoch as f32) / (epochs as f32)) * eta;

            // shuffle training data each epoch
            let mut train_data_indices: Vec<u32> = (0..train_data.0.len() as u32).collect();
            for i in 0..train_data_indices.len() {
                let j = self.series.rand_u32(i as u32, train_data.0.len() as u32) as usize;
                train_data_indices.swap(i, j);
            }

            for i in (0..train_data_indices.len()).step_by(batch_size) {
                /*
                if i % (batch_size * 500) == 0 {
                    println!(
                        "Epoch: {}/{}, Batch: {}/{}, Total Progress: {:.4}, Accuracy: {}",
                        epoch,
                        epochs,
                        i / batch_size,
                        (training_data.0.len() as f32) / (batch_size as f32),
                        ((epoch * training_data.0.len() + i) as f32)
                            / ((training_data.0.len() * epochs) as f32),
                        self.evaluate(eval_data),
                    );
                }
                */
                self.update_mini_batch(&train_data, i, i + batch_size, learning_rate, lmbda);
            }
            println!(
                "Epoch {} complete! Accuracy: {}",
                epoch,
                self.evaluate(&eval_data)
            )
        }
    }

    fn evaluate(&mut self, eval_data: &(Vec<Image>, Vec<u8>)) -> f32 {
        let accurate_results: usize = eval_data
            .0
            .iter()
            .zip(eval_data.1.iter())
            .map(|(image, label)| {
                if self.predict_label(&image) == *label {
                    1
                } else {
                    0
                }
            })
            .sum();
        accurate_results as f32 / eval_data.0.len() as f32
    }

    fn cost_derivative(&mut self, output_activations: &Vec<f32>, label: u8) -> Vec<f32> {
        let mut result = output_activations.clone();
        result[label as usize] -= 1.0;
        result
    }
}

fn main() {
    let train_images =
        load_mnist_images("mnist/train-images.idx3-ubyte", NUM_TRAIN_IMAGES).unwrap();
    let train_labels =
        load_mnist_labels("mnist/train-labels.idx1-ubyte", NUM_TRAIN_IMAGES).unwrap();
    let training_data = (train_images, train_labels);

    let eval_images = load_mnist_images("mnist/t10k-images.idx3-ubyte", NUM_EVAL_IMAGES).unwrap();
    let eval_labels = load_mnist_labels("mnist/t10k-labels.idx1-ubyte", NUM_EVAL_IMAGES).unwrap();
    let eval_data = (eval_images, eval_labels);

    let mut network = Network::new(42, &[784, 24, 10]);
    let epochs = 60;
    let batch_size = 20;
    let learning_rate = 3.0;
    let lmbda = 0.3;
    network.train_sgd(
        &training_data,
        &eval_data,
        epochs,
        batch_size,
        learning_rate,
        lmbda,
    );

    let test_image_n = 2;
    debug_print_image(&eval_data.0[test_image_n]);
    let predicted_label = network.predict_label(&eval_data.0[test_image_n]);
    println!("Actual Label: {}", eval_data.1[test_image_n]);
    println!("Predicted Label: {}", predicted_label);
}
