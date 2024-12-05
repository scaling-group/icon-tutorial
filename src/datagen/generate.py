import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import os
import h5py
from . import data_utils
from .. import utils
from .weno.weno_solver import generate_weno_scalar_sol
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def serialize_element(h5_group, equation, param, cond_k, cond_v, qoi_k, qoi_v, eqn_id, seq_id):
    '''
    equation: string describing the equation
    cond_k: condition key, 3D, (truncate, N, 1)
    cond_v: condition value, 3D, (truncate, N, 1)
    qoi_k: qoi key, 3D, (truncate, N, 1)
    qoi_v: qoi value, 3D, (truncate, N, 1)
    '''

    # Create a subgroup for each example
    example_group = h5_group.create_group(f"eid_{eqn_id}_sid_{seq_id}")
    example_group.create_dataset('equation', data=np.bytes_(equation))
    example_group.create_dataset('param', data=param)
    example_group.create_dataset('cond_k', data=cond_k)
    example_group.create_dataset('cond_v', data=cond_v)
    example_group.create_dataset('qoi_k', data=qoi_k)
    example_group.create_dataset('qoi_v', data=qoi_v)

    if (eqn_id % 100 == 0) and (seq_id == 0):
        print('-'*50, eqn_id, seq_id, '-'*50, flush=True)
        print("equation: {}".format(equation), flush=True)
        print("param: {}".format(param), flush=True)
        print("cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)

def write_evolution_hdf5(seed, eqn_type, all_params, all_eqn_ids, all_xs, all_us, stride, problem_type, file_name):
    '''
    all_xs, all_us are lists of xs and us
    xs: (N,)
    us: (num, steps + 1, N, 1), same equation, num trajectories
    write num records to the hdf5 file, each record contains truncate pairs of (cond, qoi)
    '''
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    print("===========" + file_name + "==========", flush=True)
    with h5py.File(file_name, 'w') as h5_file:
        for param, eqn_id, xs, us in zip(all_params, all_eqn_ids, all_xs, all_us):
            u1 = einshape("ijkl->(ij)kl", us[:, :-stride, :, :])  # (num * step, N, 1)
            u2 = einshape("ijkl->(ij)kl", us[:, stride:, :, :])  # (num * step, N, 1)
            
            # Shuffle the first dimension of u1 and u2, keep the same permutation
            key = next(rng)
            u1 = jax.random.permutation(key, u1, axis=0)  # (num * step, N, 1)
            u2 = jax.random.permutation(key, u2, axis=0)  # (num * step, N, 1)
            
            # Reshape u1 and u2 to (num, s, N, 1)
            u1 = einshape("(ij)kl->ijkl", u1, i=us.shape[0])  # (num, step, N, 1)
            u2 = einshape("(ij)kl->ijkl", u2, i=us.shape[0])  # (num, step, N, 1)

            if FLAGS.truncate is not None:
                u1 = u1[:, :FLAGS.truncate, :, :]  # (num, truncate, N, 1)
                u2 = u2[:, :FLAGS.truncate, :, :]  # (num, truncate, N, 1)

            x1 = einshape("k->jkl", xs, j=u1.shape[1], l=1)  # (truncate, N, 1)
            x2 = einshape("k->jkl", xs, j=u2.shape[1], l=1)  # (truncate, N, 1)

            coeff_a, coeff_b, coeff_c = param
            param_str = "{:.8f}_{:.8f}_{:.8f}".format(coeff_a, coeff_b, coeff_c)
            if problem_type == 'forward':
                for seq_id in range(us.shape[0]):
                    equation_name = f"{eqn_type}_eid{eqn_id}_sid{seq_id}_fwd_{param_str}_{stride}"
                    serialize_element(h5_file, equation=equation_name, param = param, cond_k=x1, cond_v=u1[seq_id], qoi_k=x2, qoi_v=u2[seq_id], 
                                        eqn_id = eqn_id, seq_id = seq_id)
            elif problem_type == 'backward':
                for seq_id in range(us.shape[0]):
                    equation_name = f"{eqn_type}_eid{eqn_id}_sid{seq_id}_bwd_{param_str}_{stride}"
                    serialize_element(h5_file, equation=equation_name, param = param, cond_k=x2, cond_v=u2[seq_id], qoi_k=x1, qoi_v=u1[seq_id], 
                                        eqn_id = eqn_id, seq_id = seq_id)
            else:
                raise NotImplementedError("problem_type = {} is not implemented".format(problem_type))

def generate_conservation_weno_cubic(seed, eqns, length, steps, dt, num, name, eqn_mode):
    '''du/dt + d(a * u^2 + b * u)/dx = 0'''
    eqn_type = "consv_cubic"
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    if 'random' in eqn_mode:
        minval = float(FLAGS.eqn_mode.split('_')[1])
        maxval = float(FLAGS.eqn_mode.split('_')[2])
        coeffs_a = jax.random.uniform(next(rng), shape=(eqns,), minval=minval, maxval=maxval)
        coeffs_b = jax.random.uniform(next(rng), shape=(eqns,), minval=minval, maxval=maxval)
        coeffs_c = jax.random.uniform(next(rng), shape=(eqns,), minval=minval, maxval=maxval)
    elif 'grid' in eqn_mode:
        minval = float(FLAGS.eqn_mode.split('_')[1])
        maxval = float(FLAGS.eqn_mode.split('_')[2])
        values = np.linspace(minval, maxval, eqns)
        coeffs_a, coeffs_b, coeffs_c = np.meshgrid(values, values, values)
        coeffs_a = coeffs_a.flatten()
        coeffs_b = coeffs_b.flatten()
        coeffs_c = coeffs_c.flatten()
    elif 'rlinear' in eqn_mode:
        minval = float(FLAGS.eqn_mode.split('_')[1])
        maxval = float(FLAGS.eqn_mode.split('_')[2])
        coeffs_a = jnp.zeros((eqns,))
        coeffs_b = jnp.zeros((eqns,))
        coeffs_c = jax.random.uniform(next(rng), shape=(eqns,), minval=minval, maxval=maxval)
    else:
        raise NotImplementedError("eqn_mode = {} is not implemented".format(FLAGS.eqn_mode))

    for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
        print("coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(coeff_a, coeff_b, coeff_c), flush=True)
    
    xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
    all_xs = []; all_us = []; all_params = []; all_eqn_ids = []
    for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
        print("eid{}, coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(i, coeff_a, coeff_b, coeff_c), flush=True)
        fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
        grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
        while True: # make sure the initial condition is not too large
            init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel=data_utils.rbf_circle_kernel_jax, k_sigma=1.0, k_l=1.0)[..., None]  # (num, N, 1)
            if jnp.max(jnp.abs(init)) < 3.0:
                break
        sol = generate_weno_scalar_sol(dx=1.0 / length, dt=dt, init=init, fn=fn, steps=steps, grad_fn=grad_fn, stable_tol=10.0)  # (num, steps + 1, N, 1)
        all_xs.append(xs)  # (N,)
        all_us.append(sol)  # (num, steps + 1, N, 1)
        all_params.append(jnp.array([coeff_a, coeff_b, coeff_c]))
        all_eqn_ids.append(i)
        utils.print_dot(i) # i is the index of equation
        if (i + 1) % (len(coeffs_a) // FLAGS.file_split) == 0 or i == len(coeffs_a) - 1:
            for ptype in ['forward', 'backward']:
                for st in FLAGS.stride:
                    sti = int(st)
                    write_evolution_hdf5(seed=next(rng)[0], eqn_type=eqn_type, all_params=all_params, all_eqn_ids = all_eqn_ids,
                                         all_xs=all_xs, all_us=all_us, stride=sti, problem_type=ptype, 
                                         file_name="{}_{}_{}_stride{}_{}.h5".format(name, eqn_type, ptype, sti, i + 1))
            all_xs = []; all_us = []; all_params = []; all_eqn_ids = []

def main(argv):
    for key, value in FLAGS.__flags.items():
        print(value.name, ": ", value._value, flush=True)

    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

    if not os.path.exists(FLAGS.dir):
        os.makedirs(FLAGS.dir)

    logging.info("Start!")
    generate_conservation_weno_cubic(
        seed=FLAGS.seed, eqns=FLAGS.eqns, length=FLAGS.length, steps=1000,
        dt=FLAGS.dt, num=FLAGS.num, name=name, eqn_mode=FLAGS.eqn_mode)
    logging.info("End!")

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('num', 100, 'number of records in each equation')
    flags.DEFINE_integer('eqns', 100, 'number of equations')
    flags.DEFINE_integer('length', 100, 'number of spatial points')
    flags.DEFINE_float('dt', 0.0005, 'time step in dynamics')
    flags.DEFINE_float('dx', 0.01, 'time step in dynamics')
    flags.DEFINE_string('name', 'data', 'name of the dataset')
    flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
    flags.DEFINE_list('stride', [200], 'time strides')
    flags.DEFINE_integer('seed', 1, 'random seed')
    flags.DEFINE_string('eqn_mode', 'random_-1_1', 'the mode of equation generation')
    flags.DEFINE_integer('file_split', 10, 'split the data into multiple files')
    flags.DEFINE_integer('truncate', None, 'truncate the length of each record')

    app.run(main)
