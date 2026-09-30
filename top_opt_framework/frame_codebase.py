import autograd.numpy as anp
from autograd import value_and_grad
import nlopt
import matplotlib.pyplot as plt
import problems
import topo_api

class PixelModel:
    def __init__(self, args, seed=None):
        self.seed = seed
        if seed is not None:
            anp.random.seed(seed)
        self.env = topo_api.Environment(args)
        self.nely = self.env.args['nely']
        self.nelx = self.env.args['nelx']
        self.volfrac = self.env.args['volfrac']
        self.z = anp.ones((self.nely, self.nelx)) * self.volfrac

    def get_design(self):
        """
        forward: latent z -> density field x
        """
        return self.z
    
def density_from_gaussian_beams(x_var, n_shapes, nely=50, nelx=100):
    field_total = anp.zeros(nely*nelx)
    for i in range(n_shapes):
        xi = x_var[i*5:(i+1)*5]
        w, h, theta, tx, ty = xi
        sigma = 1
        x = anp.arange(nelx)
        y = anp.arange(nely)
        xv, yv = anp.meshgrid(x, y)
        cx = tx
        cy = ty
        xt = xv - cx
        yt = yv - cy
        # xt = tx
        # yt = ty
        cos_t = anp.cos(theta)
        sin_t = anp.sin(theta)
        x_rot = cos_t * xt + sin_t * yt
        y_rot = -sin_t * xt + cos_t * yt
        dx = x_rot / (w / 2)
        dy = y_rot / (h / 2)
        r2 = dx**2 + dy**2
        # field = anp.exp(-r2 / (2 * sigma**2)) * anp.exp(-anp.maximum(0, r2 - 1.0) * 50)
        field = anp.exp(-r2 / (2 * sigma**2))
        field_total += field.ravel()
    return field_total

def ends_to_gaussian_beams(p_var, n_shapes, nely=50, nelx=100):
    x_var = []
    for i in range(n_shapes):
        xi = p_var[i*4:(i+1)*4]
        x1, y1, x2, y2 = xi
        tx = 0.5 * (x1 + x2)
        ty = 0.5 * (y1 + y2)
        dx = x2 - x1
        dy = y2 - y1
        length = anp.sqrt(dx**2 + dy**2)
        theta = anp.arctan2(-dy, dx) + 0.5 * 3.1415926
        w = 1.0
        if length == 0:
            h = 1
        else:
            h = length
        # sigma = 2.0
        x_var.extend([w, h, theta, tx, ty])

    x_var = anp.array(x_var)
    return density_from_gaussian_beams(x_var, n_shapes, nely, nelx)


def length_loss(x_var, n_shapes, target_lens):
    lengths = []
    for i in range(n_shapes):
        x1, y1, x2, y2 = x_var[4*i : 4*i+4]
        length = anp.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)
    lengths = anp.array(lengths)
    loss = anp.sum((lengths - target_lens)**2)
    return loss

def self_intersection_loss(x_var, n_shapes, nely=50, nelx=100):
    dens = ends_to_gaussian_beams(x_var, n_shapes, nely=nely, nelx=nelx)
    overflow = anp.maximum(dens - 1.0, 0.0)
    return anp.sum(overflow**2)

def joint_attaching_loss(x_var, n_shapes, target_dist=None):
    points = []
    for i in range(n_shapes):
        x1, y1, x2, y2 = x_var[4*i : 4*i+4]
        points.append(anp.array([x1, y1]))
        points.append(anp.array([x2, y2]))
    
    points = anp.stack(points)
    n_points = points.shape[0]
    
    loss = 0.0
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = anp.sqrt(anp.sum((points[i]-points[j])**2))
            if target_dist is None:
                loss += dist**2
            else:
                loss += (dist - target_dist)**2
    return loss


def run_mma(env, n_shapes=2, max_iterations=50, refresh_every=1, 
                                      max_volume_fraction=0.5, target_lens=None, x0 = None):
    nely, nelx = env.args['nely'], env.args['nelx']
    if x0 is None:
        x0 = []
        for i in range(n_shapes):
            tx1_init = anp.random.uniform(0, nelx)
            ty1_init = anp.random.uniform(0, nely)
            tx2_init = anp.random.uniform(0, nelx)
            ty2_init = anp.random.uniform(0, nely)
            x0.extend([tx1_init, ty1_init, tx2_init, ty2_init])
        x0 = anp.array(x0, dtype=anp.float64)
    

    losses = []
    frames = []

    plt.ion()
    fig, ax = plt.subplots()
    dens = ends_to_gaussian_beams(x0, n_shapes, nely=nely, nelx=nelx)
    im = ax.imshow(dens.reshape(nely, nelx), cmap='viridis', origin='lower', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    def objective_fn(x_var):
        dens = ends_to_gaussian_beams(x_var, n_shapes, nely=nely, nelx=nelx)
        return env.objective(dens, volume_constraint=False) + length_loss(x_var, n_shapes, target_lens)\
            + self_intersection_loss(x_var, n_shapes, nelx, nely)

    def wrap_autograd_func(func):
        def wrapper(x, grad_out):
            value, g = value_and_grad(func)(x)
            if grad_out.size > 0:
                grad_out[:] = g.ravel()
            dens = ends_to_gaussian_beams(x, n_shapes, nely, nelx)
            dens = anp.where(dens > 0.5, 1, dens)
            true_value = env.objective(dens, volume_constraint=False)
            losses.append(true_value)
            frames.append(anp.array(x, dtype=anp.float64))
            return value
        return wrapper

    # w, h, theta, tx, ty, sigma
    opt = nlopt.opt(nlopt.LD_MMA, 4*n_shapes)
    lower = [0,0,0,0]*n_shapes
    upper = [nelx,nely,nelx,nely]*n_shapes
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(wrap_autograd_func(objective_fn))
    opt.set_maxeval(max_iterations)
    opt.set_xtol_rel(1e-6)
    opt.set_xtol_abs(1e-8)

    x_opt = opt.optimize(x0)
    for i, x_var in enumerate(frames):
        dens = ends_to_gaussian_beams(x_var, n_shapes, nely=nely, nelx=nelx)
        im.set_data(dens.reshape(nely, nelx))
        ax.set_title(f"Iteration {i+1}, Loss: {losses[i]:.6f}")
        plt.pause(0.05)

    plt.ioff()
    plt.show()

    designs = [ends_to_gaussian_beams(f, n_shapes, nely=nely, nelx=nelx).reshape(nely, nelx) for f in frames]
    return losses, designs, frames[-1]

def generate_lengths_normal(mean, std, n, min_val, max_val):
    lengths = anp.random.normal(mean, std, size=n)
    lengths = anp.clip(lengths, min_val, max_val)
    return lengths


problem = problems.center_support(height=80, width=120)
args = topo_api.specified_task(problem)
model = PixelModel(args=args)  

nely = 100
nelx = 100

# w, h, theta, tx, ty, sigma
x0 = [40,30,40,50]

plt.ion()

fig, ax = plt.subplots()
# dens = ends_to_gaussian_beams(x0, 1, nely=nely, nelx=nelx)
# im = ax.imshow(dens.reshape(nely, nelx), cmap='viridis', origin='lower', vmin=0, vmax=1)
# fig.colorbar(im, ax=ax)
n_shapes = 15

target_lengths = generate_lengths_normal(mean=15, std=5, n=n_shapes, min_val=10, max_val=30)
print(target_lengths)
_, _, last_var = run_mma(model.env, n_shapes, 500,target_lens = target_lengths)

print(last_var)



plt.ioff()
plt.show()