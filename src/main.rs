use std::{
    cell::RefCell,
    ops::{Add, Mul, Sub},
    rc::Rc,
};

#[derive(Clone, Copy, Debug)]
enum OpType {
    // tape indices
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Tensor,
}

#[derive(Debug)]
struct Op {
    ty: OpType,
    value: f64,
    grad: f64,
    requires_grad: bool,
}

impl Op {
    fn new(ty: OpType, requires_grad: bool) -> Self {
        Self {
            ty,
            value: 0.,
            grad: 0.,
            requires_grad,
        }
    }

    fn tensor(value: f64, requires_grad: bool) -> Self {
        Self {
            ty: OpType::Tensor,
            value,
            grad: 0.,
            requires_grad,
        }
    }

    fn step(&mut self, ops: &[Op]) {
        match self.ty {
            OpType::Add(t1, t2) => self.value = ops[t1].value + ops[t2].value,
            OpType::Sub(t1, t2) => self.value = ops[t1].value - ops[t2].value,
            OpType::Mul(t1, t2) => self.value = ops[t1].value * ops[t2].value,
            OpType::Tensor => (),
        }
    }

    fn backward(&mut self, ops: &mut [Op]) {
        match self.ty {
            OpType::Add(t1, t2) => {
                if ops[t1].requires_grad {
                    ops[t1].grad += self.grad;
                };
                if ops[t2].requires_grad {
                    ops[t2].grad += self.grad;
                };
            }
            OpType::Sub(t1, t2) => {
                if ops[t1].requires_grad {
                    ops[t1].grad += self.grad;
                };
                if ops[t2].requires_grad {
                    ops[t2].grad -= self.grad;
                };
            }
            OpType::Mul(t1, t2) => {
                if ops[t1].requires_grad {
                    ops[t1].grad += self.grad * ops[t2].value;
                }
                if ops[t2].requires_grad {
                    ops[t2].grad += self.grad * ops[t1].value;
                }
            }
            OpType::Tensor => {}
        }
    }
}

#[derive(Debug)]
struct Tape {
    // add-only first
    ops: Vec<Op>,
}

impl Tape {
    fn new() -> Self {
        Self { ops: Vec::new() }
    }

    fn step(&mut self) {
        for i in 0..self.ops.len() {
            let (before, and_after) = self.ops.split_at_mut(i);
            and_after[0].step(before);
        }
    }

    fn backward(&mut self) {
        self.ops.last_mut().expect("tape shouldn't be empty").grad = 1.;
        for i in (0..self.ops.len()).rev() {
            let (before, and_after) = self.ops.split_at_mut(i);
            and_after[0].backward(before);
        }
    }
}

#[derive(Debug, Clone)]
struct Tensor {
    id: usize,
    tape: Rc<RefCell<Tape>>,
}

impl Tensor {
    fn new(value: f64, tape: Rc<RefCell<Tape>>, requires_grad: bool) -> Self {
        let mut tape_ref = tape.borrow_mut();
        let id = tape_ref.ops.len();
        tape_ref.ops.push(Op::tensor(value, requires_grad));
        drop(tape_ref);

        Self { tape, id }
    }

    fn constant(value: f64, tape: Rc<RefCell<Tape>>) -> Self {
        Self::new(value, tape, false)
    }

    fn var(value: f64, tape: Rc<RefCell<Tape>>) -> Self {
        Self::new(value, tape, true)
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut tape = self.tape.borrow_mut();
        let id = tape.ops.len();
        tape.ops.push(Op::new(OpType::Add(self.id, rhs.id), true));
        Self {
            tape: self.tape.clone(),
            id,
        }
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut tape = self.tape.borrow_mut();
        let id = tape.ops.len();
        tape.ops.push(Op::new(OpType::Mul(self.id, rhs.id), true));
        Self {
            tape: self.tape.clone(),
            id,
        }
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut tape = self.tape.borrow_mut();
        let id = tape.ops.len();
        tape.ops.push(Op::new(OpType::Sub(self.id, rhs.id), true));
        Self {
            tape: self.tape.clone(),
            id,
        }
    }
}

fn error(tape: Rc<RefCell<Tape>>, w: Tensor, b: Tensor) -> Tensor {
    // least-squares line: y = 0.263x + 3.01
    let data = [(1., 3.), (2., 4.), (5., 4.), (7., 5.)].map(|(x, y)| {
        (
            Tensor::constant(x, tape.clone()),
            Tensor::constant(y, tape.clone()),
        )
    });

    let mut err = Tensor::var(0., tape.clone());
    for (x, y) in data.into_iter() {
        err = err
            + ((w.clone() * x.clone() + b.clone() - y.clone()) * (w.clone() * x + b.clone() - y));
    }
    err
}

fn main() {
    let tape = Rc::new(RefCell::new(Tape::new()));

    let mut w_val = 2.;
    let mut b_val = 3.;
    let alpha = 0.01;

    for _ in 0..1000 {
        let w = Tensor::var(w_val, tape.clone());
        let w_id = w.id;
        let _x = Tensor::constant(4., tape.clone());
        let b = Tensor::var(b_val, tape.clone());
        let b_id = b.id;

        let err = error(tape.clone(), w, b);

        let mut t = tape.borrow_mut();
        t.step();
        t.backward();

        w_val -= alpha * t.ops[w_id].grad;
        b_val -= alpha * t.ops[b_id].grad;

        println!(
            "y = {}x + {}\t err={}",
            t.ops[w_id].value, t.ops[b_id].value, t.ops[err.id].value
        );
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
