use rand::Rng;
use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Range, Sub},
};

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

#[derive(Clone, Copy)]
enum OpType {
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    Input,
}

struct Op {
    op_type: OpType,
    value: f64,
    grad: f64,
    requires_grad: bool,
}

impl Op {
    fn new(op_type: OpType, value: f64, requires_grad: bool) -> Self {
        Self {
            op_type,
            value,
            grad: 0.,
            requires_grad,
        }
    }

    fn forward(&mut self, inputs: &[Op]) {
        use OpType::*;
        self.value = match self.op_type {
            Add(a, b) => inputs[a].value + inputs[b].value,
            Sub(a, b) => inputs[a].value - inputs[b].value,
            Mul(a, b) => inputs[a].value * inputs[b].value,
            Div(a, b) => inputs[a].value / inputs[b].value,
            Pow(a, b) => inputs[a].value.powf(inputs[b].value),
            Input => return,
        };
    }

    fn backward(&mut self, inputs: &mut [Op]) {
        use OpType::*;
        let (a, b) = match self.op_type {
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) => (a, b),
            Input => return,
        };

        let (dself_da, dself_db) = match self.op_type {
            Add(..) => (1.0, 1.0),
            Sub(..) => (1.0, -1.0),
            Mul(..) => (inputs[b].value, inputs[a].value),
            Div(..) => (
                1. / inputs[b].value,
                inputs[a].value * -1. * inputs[b].value.powf(-2.),
            ),
            Pow(..) => (
                inputs[b].value * inputs[a].value.powf(inputs[b].value - 1.),
                inputs[a].value.powf(inputs[b].value) * inputs[a].value.ln(),
            ),
            Input => unreachable!(),
        };

        if inputs[a].requires_grad {
            inputs[a].grad += self.grad * dself_da;
        }
        if inputs[b].requires_grad {
            inputs[b].grad += self.grad * dself_db;
        }
    }
}

struct Tape {
    ops: Vec<Op>,
}

impl Tape {
    fn new() -> Self {
        Self { ops: Vec::new() }
    }

    fn zero_grad(&mut self) {
        for op in &mut self.ops {
            op.grad = 0.;
        }
    }

    fn forward(&mut self) {
        for i in 0..self.ops.len() {
            let (inputs, [current, ..]) = self.ops.split_at_mut(i) else {
                unreachable!()
            };
            current.forward(inputs);
        }
    }

    fn backward(&mut self) {
        if let Some(output) = self.ops.last_mut() {
            output.grad = 1.;
        }
        for i in (0..self.ops.len()).rev() {
            let (inputs, [current, ..]) = self.ops.split_at_mut(i) else {
                unreachable!()
            };
            current.backward(inputs);
        }
    }

    fn requires_grad(&self, tensor: Tensor) -> bool {
        self.ops[tensor.id].requires_grad
    }

    fn value(&self, tensor: Tensor) -> f64 {
        self.ops[tensor.id].value
    }

    fn grad(&self, tensor: Tensor) -> f64 {
        self.ops[tensor.id].grad
    }

    fn value_mut(&mut self, tensor: Tensor) -> &mut f64 {
        &mut self.ops[tensor.id].value
    }
}

#[derive(Clone, Copy)]
struct Tensor {
    id: usize,
}

impl Tensor {
    fn new(value: f64, requires_grad: bool) -> Self {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            let id = tape.ops.len();
            tape.ops.push(Op::new(OpType::Input, value, requires_grad));
            Self { id }
        })
    }

    fn constant(value: f64) -> Self {
        Self::new(value, false)
    }

    fn var(value: f64) -> Self {
        Self::new(value, true)
    }

    fn rand(range: Range<f64>) -> Self {
        Self::var(rand::rng().random_range(range))
    }

    fn binary_op(self, rhs: Self, op_ctor: impl Fn(usize, usize) -> OpType) -> Self {
        TAPE.with(|t| {
            let mut t = t.borrow_mut();
            let requires_grad = t.requires_grad(self) || t.requires_grad(rhs);
            let id = t.ops.len();
            t.ops
                .push(Op::new(op_ctor(self.id, rhs.id), 0.0, requires_grad));
            Self { id }
        })
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, OpType::Add)
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, OpType::Mul)
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, OpType::Sub)
    }
}

impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, OpType::Div)
    }
}

pub trait Pow<Rhs = Self> {
    type Output;

    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl Pow for Tensor {
    type Output = Self;
    fn pow(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, OpType::Pow)
    }
}

fn linear_regression_mse(
    data: impl IntoIterator<Item = (f64, f64)>,
    w: Tensor,
    b: Tensor,
) -> Tensor {
    let square = Tensor::constant(2.);
    let mut err = Tensor::var(0.);
    for (x_val, y_val) in data {
        let x = Tensor::constant(x_val);
        let y = Tensor::constant(y_val);
        let diff = w * x + b - y;
        err = err + diff.pow(square);
    }
    err
}

fn main() {
    // least-squares line: y = 0.264x + 3.011
    let data = [(1., 3.), (2., 4.), (5., 4.), (7., 5.)];
    let alpha = 0.01;

    let w = Tensor::rand(-1. ..1.);
    let b = Tensor::rand(-1. ..1.);
    let err = linear_regression_mse(data, w, b);

    TAPE.with(|t| {
        for _ in 0..1000 {
            let mut t = t.borrow_mut();

            t.zero_grad();
            t.forward();
            t.backward();

            *t.value_mut(w) -= alpha * t.grad(w);
            *t.value_mut(b) -= alpha * t.grad(b);

            println!("y = {}x + {}, err={}", t.value(w), t.value(b), t.value(err));
        }
    });
}
