import autograd.numpy as anp
from autograd import value_and_grad
import nlopt
import matplotlib.pyplot as plt
import problems
import topo_api
import pulp


# =========================
# Utilities
# =========================

def extract_nodes_and_edges(x0):
    nodes = []
    edges = []
    lens = []
    node_map = {}

    def get_node_index(pt):
        key = (pt[0], pt[1])
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append(pt)
        return node_map[key]

    for i in range(len(x0)//4):
        x1,y1,x2,y2 = x0[4*i:4*i+4]
        i1 = get_node_index((x1,y1))
        i2 = get_node_index((x2,y2))
        edges.append((i1, i2))
        lens.append(anp.sqrt((x1-x2)**2 + (y1-y2)**2))

    return anp.array(nodes).ravel(), edges, lens


# =========================
# Density functions
# =========================

def density_from_gaussian_beams(x_var, n_shapes, nely=50, nelx=100):
    field_total = anp.zeros(nely*nelx)

    x = anp.arange(nelx)
    y = anp.arange(nely)
    xv, yv = anp.meshgrid(x, y)

    for i in range(n_shapes):
        w, h, theta, tx, ty = x_var[i*5:(i+1)*5]

        xt = xv - tx
        yt = yv - ty

        cos_t = anp.cos(theta)
        sin_t = anp.sin(theta)

        x_rot = cos_t * xt + sin_t * yt
        y_rot = -sin_t * xt + cos_t * yt

        dx = x_rot / (w / 2)
        dy = y_rot / (h / 2)

        r2 = dx**2 + dy**2
        field = anp.exp(-r2 / 2.0)

        field_total += field.ravel()

    return field_total


def points_to_gaussian_beams(p_var, edges, nely=50, nelx=100):
    n_points = len(p_var) // 2
    points = p_var.reshape((n_points, 2))

    x_var = []

    for (i, j) in edges:
        x1, y1 = points[i]
        x2, y2 = points[j]

        tx = 0.5 * (x1 + x2)
        ty = 0.5 * (y1 + y2)

        dx = x2 - x1
        dy = y2 - y1

        length = anp.sqrt(dx**2 + dy**2)
        theta = -anp.arctan2(dx, dy)

        w = 1.0
        h = anp.where(length > 1e-6, length * 0.75, 1.0)

        x_var.extend([w, h, theta, tx, ty])

    x_var = anp.array(x_var)
    return density_from_gaussian_beams(x_var, len(edges), nely, nelx)


# =========================
# Losses
# =========================

def length_loss_points(p_var, edges, target_lens):
    points = p_var.reshape((-1, 2))
    loss = 0.0

    for k, (i, j) in enumerate(edges):
        x1, y1 = points[i]
        x2, y2 = points[j]
        length = anp.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        loss += (length - target_lens[k])**2

    return loss


def self_intersection_loss_points(p_var, edges, nely, nelx):
    dens = points_to_gaussian_beams(p_var, edges, nely, nelx)
    overflow = anp.maximum(dens - 1.0, 0.0)
    return anp.sum((overflow+1)**4)


# =========================
# Optimization
# =========================

def run_mma(env, p0, edges, max_iterations=200, target_lens=None):
    nely, nelx = env.args['nely'], env.args['nelx']

    losses = []
    frames = []

    plt.ion()
    fig, ax = plt.subplots()

    dens = points_to_gaussian_beams(p0, edges, nely, nelx)
    im = ax.imshow(dens.reshape(nely, nelx), cmap='viridis', origin='lower', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    def objective_fn(p_var):
        dens = points_to_gaussian_beams(p_var, edges, nely, nelx)
        return (
            env.objective(dens, volume_constraint=False)
            + length_loss_points(p_var, edges, target_lens)
            + self_intersection_loss_points(p_var, edges, nely, nelx)
        )

    def wrapper(x, grad_out):
        value, g = value_and_grad(objective_fn)(x)
        if grad_out.size > 0:
            grad_out[:] = g

        dens = points_to_gaussian_beams(x, edges, nely, nelx)
        losses.append(env.objective(dens, volume_constraint=False))
        frames.append(anp.array(x))

        return value

    opt = nlopt.opt(nlopt.LD_MMA, len(p0))

    n_nodes = len(p0) // 2
    lower = [0,0] * n_nodes
    upper = [nelx, nely] * n_nodes

    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(wrapper)
    opt.set_maxeval(max_iterations)
    # opt.set_initial_step(0.001)

    p_opt = opt.optimize(p0)

    # visualization
    # scatter_points = []

    # for i, p_var in enumerate(frames):
    #     dens = points_to_gaussian_beams(p_var, edges, nely, nelx)
    #     im.set_data(dens.reshape(nely, nelx))

    #     points = p_var.reshape((-1, 2))

    #     for sp in scatter_points:
    #         sp.remove()
    #     scatter_points = []

    #     for (i1, i2) in edges:
    #         x1,y1 = points[i1]
    #         x2,y2 = points[i2]
    #         sp = ax.plot([x1,x2],[y1,y2],'r-')[0]
    #         scatter_points.append(sp)

    #     ax.set_title(f"Iter {i+1}, Loss {losses[i]:.4f}")
    #     plt.pause(0.05)

    # plt.ioff()
    # plt.show()

    return losses, frames, p_opt


def assign_inventory_to_edges(points, edges, inventory_lengths):
    points = points.reshape((-1, 2))
    n_edges = len(edges)
    num_inventory = len(inventory_lengths)

    member_lengths = anp.array([
        anp.linalg.norm(points[i] - points[j]) for i,j in edges
    ])

    prob = pulp.LpProblem("InventoryAssignment", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x",
        ((i, j) for i in range(n_edges) for j in range(num_inventory)),
        cat="Binary"
    )

    prob += pulp.lpSum(
        x[(i, j)] * anp.abs(inventory_lengths[j] - member_lengths[i])
        for i in range(n_edges)
        for j in range(num_inventory)
    )

    for i in range(n_edges):
        prob += pulp.lpSum(x[(i, j)] for j in range(num_inventory)) == 1

    # for i in range(n_edges):
    #     for j in range(num_inventory):
    #         if inventory_lengths[j] < member_lengths[i]:
    #             prob += x[(i, j)] == 0

    for j in range(num_inventory):
        prob += pulp.lpSum(x[(i, j)] for i in range(n_edges)) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    assignment_dict = {}
    target_lengths = anp.zeros(n_edges)
    for i in range(n_edges):
        for j in range(num_inventory):
            if pulp.value(x[(i, j)]) == 1:
                assignment_dict[i] = j
                target_lengths[i] = inventory_lengths[j]

    return target_lengths, assignment_dict


# =========================
# Main
# =========================

# problem = problems.example(height=80, width=120)
problem = problems.example(height=60, width=120)
args = topo_api.specified_task(problem)
env = topo_api.Environment(args)

# original beams
# x0 = [
#         0, 40, 12, 52,
#         0, 40, 24, 40,
#         24, 40, 12, 52,
#         12, 52, 36, 52,
#         24, 40, 36, 52,
#         24, 40, 48, 40,
#         36, 52, 48, 40,
#         36, 52, 60, 52,
#         48, 40, 60, 52,
#         48, 40, 72, 40,
#         72, 40, 60, 52,
#         60, 52, 84, 52,
#         72, 40, 84, 52,
#         72, 40, 96, 40,
#         84, 52, 96, 40,
#         84, 52, 108, 52,
#         96, 40, 108, 52,
#         96, 40, 120, 40,
#         108, 52, 120, 40 
#     ]

# x0 = [
#         30, 20, 30, 50,
#         30, 20, 38, 45,
#         30, 50, 38, 45,
#         38, 45, 40, 50,
#         40, 50, 30, 50,
#         30, 50, 30, 60,
#         30, 50, 40, 60,
#         40, 50, 40, 60,
#         30, 60, 40, 60,
#         30, 80, 40, 72,
#         30, 60, 30, 70,
#         30, 70, 40, 60,
#         40, 60, 40, 72,
#         30, 70, 40, 72,
#         30, 70, 30, 80,
#         30, 80, 40, 83,
#         40, 72, 40, 83,
#         40, 60, 50, 74,
#         40, 72, 50, 74, 
#         40, 83, 50, 74,
#         40, 83, 50, 86,
#         50, 74, 50, 86,
#         50, 74, 60, 76,
#         50, 86, 60, 76,
#         50, 86, 60, 89,
#         60, 89, 70, 92,
#         60, 89, 60, 76,
#         60, 76, 70, 92
#     ]


x0 = [
    0, 59, 76, 58,
    0, 59, 54, 30,
    54, 30, 76, 58,
    76, 58, 119, 30,
    0, 0, 54, 30,
    0, 0, 76, 4,
    54, 30, 76, 4, 
    76, 4, 119, 30
]


size = 20
lam = 1.0
poisson_vals = anp.random.poisson(lam=lam, size=size)
min_val, max_val = 0.7*40, 2.0*40
scaled_vals = min_val + (poisson_vals - poisson_vals.min()) / (poisson_vals.max() - poisson_vals.min()) * (max_val - min_val)
inventory_lens = scaled_vals
p, edges, lens = extract_nodes_and_edges(x0)
nely, nelx = env.args['nely'], env.args['nelx']

fig, ax = plt.subplots()
dens = points_to_gaussian_beams(p, edges, nely, nelx)
im = ax.imshow(dens.reshape(nely, nelx), cmap='viridis', origin='lower', vmin=0, vmax=1)
fig.colorbar(im, ax=ax)
dens = points_to_gaussian_beams(p, edges, nely, nelx)
im.set_data(dens.reshape(nely, nelx))
points = p.reshape((-1, 2))
scatter_points = []
for (i1, i2) in edges:
    x1,y1 = points[i1]
    x2,y2 = points[i2]
    sp = ax.plot([x1,x2],[y1,y2],'r-')[0]
    scatter_points.append(sp)
fig.savefig(f"cmp_truss2_-1.png", dpi=300, bbox_inches='tight')
plt.close(fig)

for i in range(10):
    _, _, p = run_mma(env, p, edges, max_iterations=200, target_lens=lens)
    lens, _ = assign_inventory_to_edges(p, edges, inventory_lens)

    fig, ax = plt.subplots()
    dens = points_to_gaussian_beams(p, edges, nely, nelx)
    im = ax.imshow(dens.reshape(nely, nelx), cmap='viridis', origin='lower', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    dens = points_to_gaussian_beams(p, edges, nely, nelx)
    im.set_data(dens.reshape(nely, nelx))
    points = p.reshape((-1, 2))
    scatter_points = []
    for (i1, i2) in edges:
        x1,y1 = points[i1]
        x2,y2 = points[i2]
        sp = ax.plot([x1,x2],[y1,y2],'r-')[0]
        scatter_points.append(sp)
    print(f"compliance {env.objective(dens)}")
    fig.savefig(f"cmp_truss2_{i}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"itr {i} finished")


