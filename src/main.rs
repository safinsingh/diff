use num_traits::Pow;
use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
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
            Pow(a, b) => inputs[a].value.pow(inputs[b].value),
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
                inputs[a].value * -1. * inputs[b].value.pow(-2),
            ),
            Pow(..) => (
                inputs[b].value * inputs[a].value.pow(inputs[b].value - 1.),
                inputs[a].value.pow(inputs[b].value) * inputs[a].value.ln(),
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

    fn clear(&mut self) {
        self.ops.clear();
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
}

#[derive(Clone, Copy)]
struct Tensor {
    id: usize,
}

impl Tensor {
    fn constant(value: f64) -> Self {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            let id = tape.ops.len();
            tape.ops.push(Op::new(OpType::Input, value, false));
            Self { id }
        })
    }

    fn var(value: f64) -> Self {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            let id = tape.ops.len();
            tape.ops.push(Op::new(OpType::Input, value, true));
            Self { id }
        })
    }

    fn binary_op(self, rhs: Self, op_ctor: impl Fn(usize, usize) -> OpType) -> Self {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            let requires_grad = tape.ops[self.id].requires_grad || tape.ops[rhs.id].requires_grad;
            let id = tape.ops.len();
            tape.ops
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

impl num_traits::Pow<Tensor> for Tensor {
    type Output = Self;
    fn pow(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, OpType::Pow)
    }
}

fn linear_regression_mse(data: impl Iterator<Item = (f64, f64)>, w: Tensor, b: Tensor) -> Tensor {
    let mut err = Tensor::constant(0.);
    for (x_val, y_val) in data {
        let x = Tensor::constant(x_val);
        let y = Tensor::constant(y_val);
        let diff = w * x + b - y;
        err = err + diff.pow(Tensor::constant(2.));
    }
    err
}

fn main() {
    let data = [(1., 3.), (2., 4.), (5., 4.), (7., 5.)];

    let mut w_val = 2.;
    let mut b_val = 3.;
    let alpha = 0.01;

    for _ in 0..1000 {
        TAPE.with(|t| t.borrow_mut().clear());

        let w = Tensor::var(w_val);
        let b = Tensor::var(b_val);
        let err = linear_regression_mse(data.into_iter(), w, b);

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            tape.forward();
            tape.backward();

            w_val -= alpha * tape.ops[w.id].grad;
            b_val -= alpha * tape.ops[b.id].grad;

            // least-squares line: y = 0.263x + 3.01
            println!(
                "y = {}x + {}\t err={}",
                tape.ops[w.id].value, tape.ops[b.id].value, tape.ops[err.id].value
            );
        });
    }

    // f(w, b) = wx + b
    // f(2, 3) = 2x + 3
    //
    // at x=4:
    //
    // df/d(wx+b) = 1
    // df/d(wx) = 1
    // df/dw = 4
    // df/db = 1
    // df/dx = 0
}
